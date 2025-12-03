"""
Retrieval module for AMR-Graph.

This module implements the three-stage retrieval pipeline:
1. Anchor Retrieval (BM25 + FAISS fusion)
2. Graph Navigation (multi-channel BFS/beam search)
3. Verification (LLM relevance scoring and filtering)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

from config import get_pipeline_config
from api_utils import (
    call_llm_with_retry,
    build_text_message,
    get_embedding_with_retry,
    parse_json_response,
    parse_json_list_response
)
from prompts import (
    PROMPT_QUESTION_TYPING,
    PROMPT_RELEVANCE_SCORING,
    PROMPT_FINAL_ANSWER,
    PROMPT_ANSWER_WITH_REASONING,
    format_events_for_context
)
from event_node import AdaptiveEventNode
from amr_graph import AMRGraph
from indexing import UnifiedIndex


logger = logging.getLogger(__name__)


# Channel priorities by question type
CHANNEL_PRIORITIES = {
    "Object_Tracking": ["object_lifecycle", "temporal", "entity_jump"],
    "Person_Understanding": ["entity_jump", "temporal", "object_lifecycle"],
    "Temporal_Multihop": ["temporal", "entity_jump", "object_lifecycle"],
    "Cross_Modal": ["temporal", "entity_jump", "object_lifecycle"],
    "General": ["temporal", "entity_jump", "object_lifecycle"],
}


@dataclass
class QuestionAnalysis:
    """Analysis result for a question."""
    question_type: str
    key_entities: List[str]
    reasoning: str


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline."""
    anchor_nodes: List[int]
    navigated_nodes: List[int]
    verified_nodes: List[int]
    context_text: str
    final_answer: str
    metadata: Dict[str, Any]


class QuestionAnalyzer:
    """
    Analyzes questions to determine type and extract key entities.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize question analyzer.
        
        Args:
            model_name: LLM model for question analysis.
        """
        config = get_pipeline_config()
        self.model_name = model_name or config.llm_model_question_typing
    
    def analyze(self, question: str) -> QuestionAnalysis:
        """
        Analyze a question to determine its type and key entities.
        
        Args:
            question: The question to analyze.
            
        Returns:
            QuestionAnalysis with type and entities.
        """
        prompt = PROMPT_QUESTION_TYPING.format(question=question)
        
        try:
            response, _ = call_llm_with_retry(
                model_name=self.model_name,
                messages=[build_text_message(prompt)],
                temperature=0.0,
                max_tokens=500
            )
            
            result = parse_json_response(response)
            
            return QuestionAnalysis(
                question_type=result.get("question_type", "General"),
                key_entities=result.get("key_entities", []),
                reasoning=result.get("reasoning_brief", "")
            )
            
        except Exception as e:
            logger.warning(f"Question analysis failed: {e}, defaulting to General")
            return QuestionAnalysis(
                question_type="General",
                key_entities=[],
                reasoning=""
            )


class AnchorRetriever:
    """
    Stage 1: Anchor retrieval using hybrid sparse/dense search.
    """
    
    def __init__(
        self,
        unified_index: UnifiedIndex,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize anchor retriever.
        
        Args:
            unified_index: Unified search index.
            embedding_model: Model for query embeddings.
        """
        self.index = unified_index
        config = get_pipeline_config()
        self.embedding_model = embedding_model or config.embedding_model
        self._config = config
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve anchor nodes for a query.
        
        Args:
            query: Search query.
            top_k: Number of anchors to retrieve.
            
        Returns:
            List of (node_id, score) tuples.
        """
        if top_k is None:
            top_k = self._config.anchor_fusion_k
        
        # Get query embedding
        try:
            embeddings = get_embedding_with_retry(self.embedding_model, query)
            query_embedding = np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}")
            query_embedding = None
        
        # Hybrid search
        results = self.index.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        return results


class GraphNavigator:
    """
    Stage 2: Graph navigation using multi-channel BFS/beam search.
    """
    
    def __init__(self, graph: AMRGraph, embedding_model: Optional[str] = None):
        """
        Initialize graph navigator.
        
        Args:
            graph: AMR-Graph to navigate.
            embedding_model: Model for similarity scoring.
        """
        self.graph = graph
        config = get_pipeline_config()
        self.embedding_model = embedding_model or config.embedding_model
        self._config = config
    
    def navigate(
        self,
        anchor_nodes: List[int],
        question_type: str,
        query_embedding: Optional[np.ndarray] = None,
        max_hops: Optional[int] = None,
        beam_width: Optional[int] = None,
        max_visited: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Navigate the graph from anchor nodes to find relevant events.
        
        Args:
            anchor_nodes: Starting node IDs.
            question_type: Type of question for channel prioritization.
            query_embedding: Query embedding for relevance scoring.
            max_hops: Maximum traversal depth.
            beam_width: Beam width for search.
            max_visited: Maximum nodes to visit.
            
        Returns:
            List of (node_id, score) tuples.
        """
        if max_hops is None:
            max_hops = self._config.navigation_max_hops
        if beam_width is None:
            beam_width = self._config.navigation_beam_width
        if max_visited is None:
            max_visited = self._config.max_visited_nodes
        
        # Get channel priorities
        channels = CHANNEL_PRIORITIES.get(question_type, CHANNEL_PRIORITIES["General"])
        
        # Initialize frontier with anchors
        visited: Set[int] = set()
        frontier: Dict[int, float] = {nid: 1.0 for nid in anchor_nodes if nid in self.graph.nodes}
        
        all_scores: Dict[int, float] = dict(frontier)
        
        for hop in range(max_hops):
            if not frontier or len(visited) >= max_visited:
                break
            
            next_frontier: Dict[int, float] = {}
            
            for node_id, current_score in frontier.items():
                if node_id in visited:
                    continue
                
                visited.add(node_id)
                
                # Get outgoing edges
                edges = self.graph.get_outgoing_edges(node_id, channels)
                
                # Score neighbors by channel priority
                for channel_idx, channel in enumerate(channels):
                    channel_weight = 1.0 / (channel_idx + 1)  # Higher weight for priority channels
                    
                    for dst_id, payload in edges.get(channel, []):
                        if dst_id in visited:
                            continue
                        
                        # Calculate neighbor score
                        neighbor_score = current_score * channel_weight * 0.8
                        
                        # Add similarity score if embedding available
                        if query_embedding is not None and dst_id in self.graph.nodes:
                            dst_node = self.graph.nodes[dst_id]
                            dst_embedding = dst_node.get_embedding_array()
                            if dst_embedding is not None:
                                similarity = self._cosine_similarity(query_embedding, dst_embedding)
                                neighbor_score *= (1 + similarity) / 2
                        
                        # Update scores
                        if dst_id not in next_frontier or next_frontier[dst_id] < neighbor_score:
                            next_frontier[dst_id] = neighbor_score
                        
                        if dst_id not in all_scores or all_scores[dst_id] < neighbor_score:
                            all_scores[dst_id] = neighbor_score
            
            # Beam selection
            sorted_frontier = sorted(
                next_frontier.items(),
                key=lambda x: x[1],
                reverse=True
            )
            frontier = dict(sorted_frontier[:beam_width])
        
        # Return all visited nodes with scores
        results = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:max_visited]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))


class ResultVerifier:
    """
    Stage 3: Verify and filter retrieved results using LLM.
    """
    
    def __init__(
        self,
        graph: AMRGraph,
        model_name: Optional[str] = None
    ):
        """
        Initialize result verifier.
        
        Args:
            graph: AMR-Graph for node access.
            model_name: LLM model for verification.
        """
        self.graph = graph
        config = get_pipeline_config()
        self.model_name = model_name or config.llm_model_verification
    
    def verify(
        self,
        candidate_nodes: List[int],
        question: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Verify and score candidate nodes for relevance.
        
        Args:
            candidate_nodes: Node IDs to verify.
            question: The question being answered.
            top_k: Number of top results to keep.
            
        Returns:
            List of (node_id, relevance_score) tuples.
        """
        if not candidate_nodes:
            return []
        
        # Sort nodes by time for context
        sorted_nodes = sorted(
            candidate_nodes,
            key=lambda nid: self.graph.nodes[nid].time_start
        )
        
        # Format events for LLM
        events = []
        for nid in sorted_nodes:
            node = self.graph.nodes[nid]
            events.append({
                "node_id": nid,
                "time_start": node.time_start,
                "time_end": node.time_end,
                "summary_text": node.summary_text
            })
        
        events_text = format_events_for_context(events)
        
        prompt = PROMPT_RELEVANCE_SCORING.format(
            question=question,
            events=events_text
        )
        
        try:
            response, _ = call_llm_with_retry(
                model_name=self.model_name,
                messages=[build_text_message(prompt)],
                temperature=0.0,
                max_tokens=500
            )
            
            scores = parse_json_list_response(response)
            
            # Pair scores with node IDs
            results = []
            for i, score in enumerate(scores):
                if i < len(sorted_nodes):
                    try:
                        score_val = float(score)
                        results.append((sorted_nodes[i], score_val))
                    except (ValueError, TypeError):
                        results.append((sorted_nodes[i], 0.0))
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.warning(f"Verification failed: {e}, returning candidates as-is")
            # Return candidates with default scores
            return [(nid, 0.5) for nid in sorted_nodes[:top_k]]


class AnswerGenerator:
    """
    Generate final answers based on retrieved context.
    """
    
    def __init__(
        self,
        graph: AMRGraph,
        model_name: Optional[str] = None
    ):
        """
        Initialize answer generator.
        
        Args:
            graph: AMR-Graph for node access.
            model_name: LLM model for answer generation.
        """
        self.graph = graph
        config = get_pipeline_config()
        self.model_name = model_name or config.llm_model_final_answer
    
    def generate(
        self,
        verified_nodes: List[int],
        question: str,
        include_reasoning: bool = False
    ) -> str:
        """
        Generate answer based on verified context.
        
        Args:
            verified_nodes: Node IDs forming the context.
            question: The question to answer.
            include_reasoning: Whether to include reasoning in response.
            
        Returns:
            Generated answer string.
        """
        if not verified_nodes:
            return "I could not find relevant information to answer this question."
        
        # Sort by time
        sorted_nodes = sorted(
            verified_nodes,
            key=lambda nid: self.graph.nodes[nid].time_start
        )
        
        # Build context
        context_parts = []
        for nid in sorted_nodes:
            node = self.graph.nodes[nid]
            time_str = f"[{node.time_start:.1f}s - {node.time_end:.1f}s]"
            context_parts.append(f"{time_str} {node.summary_text}")
            
            # Add key dialogue if present
            if node.dialogue_snippets:
                dialogue_str = " | ".join(node.dialogue_snippets[:3])
                context_parts.append(f"  Dialogue: {dialogue_str}")
        
        context_text = "\n".join(context_parts)
        
        # Select prompt
        if include_reasoning:
            prompt = PROMPT_ANSWER_WITH_REASONING.format(
                question=question,
                context=context_text
            )
        else:
            prompt = PROMPT_FINAL_ANSWER.format(
                question=question,
                context=context_text
            )
        
        try:
            response, _ = call_llm_with_retry(
                model_name=self.model_name,
                messages=[build_text_message(prompt)],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Failed to generate answer due to an error."


class RetrievalPipeline:
    """
    Complete retrieval pipeline combining all stages.
    """
    
    def __init__(
        self,
        graph: AMRGraph,
        unified_index: UnifiedIndex
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            graph: AMR-Graph to search.
            unified_index: Unified search index.
        """
        self.graph = graph
        self.index = unified_index
        self._config = get_pipeline_config()
        
        # Initialize components
        self.question_analyzer = QuestionAnalyzer()
        self.anchor_retriever = AnchorRetriever(unified_index)
        self.navigator = GraphNavigator(graph)
        self.verifier = ResultVerifier(graph)
        self.answer_generator = AnswerGenerator(graph)
    
    def retrieve_and_answer(
        self,
        question: str,
        anchor_k: Optional[int] = None,
        verify_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Run the complete retrieval pipeline and generate an answer.
        
        Args:
            question: The question to answer.
            anchor_k: Number of anchor nodes.
            verify_k: Number of verified nodes to keep.
            
        Returns:
            RetrievalResult with all intermediate and final results.
        """
        if anchor_k is None:
            anchor_k = self._config.anchor_fusion_k
        if verify_k is None:
            verify_k = 10
        
        metadata = {"question": question}
        
        # Stage 0: Question analysis
        analysis = self.question_analyzer.analyze(question)
        metadata["question_type"] = analysis.question_type
        metadata["key_entities"] = analysis.key_entities
        
        logger.info(f"Question type: {analysis.question_type}")
        
        # Get query embedding
        try:
            embeddings = get_embedding_with_retry(
                self._config.embedding_model,
                question
            )
            query_embedding = np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}")
            query_embedding = None
        
        # Stage 1: Anchor retrieval
        anchors = self.anchor_retriever.retrieve(question, anchor_k)
        anchor_nodes = [nid for nid, _ in anchors]
        
        logger.info(f"Retrieved {len(anchor_nodes)} anchor nodes")
        metadata["anchor_scores"] = dict(anchors)
        
        # Stage 2: Graph navigation
        navigated = self.navigator.navigate(
            anchor_nodes,
            analysis.question_type,
            query_embedding
        )
        navigated_nodes = [nid for nid, _ in navigated]
        
        logger.info(f"Navigation found {len(navigated_nodes)} candidate nodes")
        metadata["navigation_scores"] = dict(navigated)
        
        # Stage 3: Verification
        verified = self.verifier.verify(navigated_nodes, question, verify_k)
        verified_nodes = [nid for nid, _ in verified]
        
        logger.info(f"Verified {len(verified_nodes)} relevant nodes")
        metadata["verification_scores"] = dict(verified)
        
        # Build context text
        sorted_verified = sorted(
            verified_nodes,
            key=lambda nid: self.graph.nodes[nid].time_start
        )
        context_parts = []
        for nid in sorted_verified:
            node = self.graph.nodes[nid]
            context_parts.append(node.summary_text)
        context_text = " | ".join(context_parts)
        
        # Generate final answer
        final_answer = self.answer_generator.generate(verified_nodes, question)
        
        return RetrievalResult(
            anchor_nodes=anchor_nodes,
            navigated_nodes=navigated_nodes,
            verified_nodes=verified_nodes,
            context_text=context_text,
            final_answer=final_answer,
            metadata=metadata
        )


if __name__ == "__main__":
    # Test retrieval components
    from event_node import EventNodeFactory
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test graph and index
    factory = EventNodeFactory()
    graph = AMRGraph(video_id="test_video")
    index = UnifiedIndex()
    
    # Create test nodes
    test_events = [
        {
            "summary": "Lily enters the kitchen and picks up a coffee mug",
            "persons": ["Lily"],
            "objects": ["coffee mug", "kitchen"],
            "dialogue": ["I need my morning coffee"],
            "actions": [{"actor": "Lily", "verb": "pick up", "object": "coffee mug"}]
        },
        {
            "summary": "Lily makes mocha coffee at the counter",
            "persons": ["Lily"],
            "objects": ["coffee", "counter", "mocha"],
            "dialogue": ["Mocha is my favorite"],
            "actions": [{"actor": "Lily", "verb": "make", "object": "mocha coffee"}]
        },
        {
            "summary": "Emma walks in and greets Lily",
            "persons": ["Emma", "Lily"],
            "objects": ["door"],
            "dialogue": ["Good morning Lily!", "Hey Emma!"],
            "actions": [{"actor": "Emma", "verb": "greet", "object": "Lily"}]
        },
    ]
    
    for i, data in enumerate(test_events):
        node = AdaptiveEventNode(
            node_id=factory.allocate_id(),
            video_id="test_video",
            clip_ids=[f"clip_{i}"],
            time_start=float(i * 30),
            time_end=float((i + 1) * 30),
            summary_text=data["summary"],
            persons=data["persons"],
            objects=data["objects"],
            dialogue_snippets=data["dialogue"],
            actions=data["actions"],
        )
        graph.process_new_node(node)
        index.add_node(node)
    
    # Test question analyzer
    print("Testing Question Analyzer:")
    analyzer = QuestionAnalyzer()
    analysis = analyzer.analyze("What kind of coffee does Lily like?")
    print(f"  Type: {analysis.question_type}")
    print(f"  Entities: {analysis.key_entities}")
    
    # Test anchor retrieval
    print("\nTesting Anchor Retrieval:")
    retriever = AnchorRetriever(index)
    anchors = retriever.retrieve("Lily coffee", top_k=3)
    print(f"  Anchors: {anchors}")
    
    print("\nAll tests completed!")
