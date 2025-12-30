"""
Indexing module for AMR-Graph retrieval.

This module implements BM25 sparse indexing, FAISS dense indexing,
and provides unified retrieval interfaces with incremental update support.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
import math
import re

from event_node import AdaptiveEventNode
from config import get_pipeline_config
from api_utils import get_embedding_with_retry


logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 sparse retrieval index with incremental update support.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        
        # Document storage: doc_id -> tokenized document
        self.documents: Dict[int, List[str]] = {}
        
        # Inverted index: term -> set of doc_ids
        self.inverted_index: Dict[str, Set[int]] = {}
        
        # Document frequencies: term -> count of docs containing term
        self.doc_frequencies: Dict[str, int] = {}
        
        # Document lengths
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text.
            
        Returns:
            List of lowercase tokens.
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def add_document(self, doc_id: int, text: str) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Document identifier.
            text: Document text.
        """
        tokens = self._tokenize(text)
        
        if doc_id in self.documents:
            # Remove old document first
            self.remove_document(doc_id)
        
        self.documents[doc_id] = tokens
        self.doc_lengths[doc_id] = len(tokens)
        
        # Update inverted index and doc frequencies
        seen_terms = set()
        for token in tokens:
            if token not in self.inverted_index:
                self.inverted_index[token] = set()
            self.inverted_index[token].add(doc_id)
            
            if token not in seen_terms:
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1
                seen_terms.add(token)
        
        # Update statistics
        self.total_docs += 1
        self._update_avg_length()
    
    def remove_document(self, doc_id: int) -> None:
        """
        Remove a document from the index.
        
        Args:
            doc_id: Document identifier.
        """
        if doc_id not in self.documents:
            return
        
        tokens = self.documents[doc_id]
        seen_terms = set()
        
        for token in tokens:
            if token in self.inverted_index:
                self.inverted_index[token].discard(doc_id)
                if not self.inverted_index[token]:
                    del self.inverted_index[token]
            
            if token not in seen_terms:
                if token in self.doc_frequencies:
                    self.doc_frequencies[token] -= 1
                    if self.doc_frequencies[token] <= 0:
                        del self.doc_frequencies[token]
                seen_terms.add(token)
        
        del self.documents[doc_id]
        del self.doc_lengths[doc_id]
        self.total_docs -= 1
        self._update_avg_length()
    
    def _update_avg_length(self) -> None:
        """Update average document length."""
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        else:
            self.avg_doc_length = 0.0
    
    def _idf(self, term: str) -> float:
        """
        Calculate IDF for a term.
        
        Args:
            term: The term.
            
        Returns:
            IDF value.
        """
        df = self.doc_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
    
    def _score_document(self, doc_id: int, query_tokens: List[str]) -> float:
        """
        Calculate BM25 score for a document given query.
        
        Args:
            doc_id: Document identifier.
            query_tokens: Tokenized query.
            
        Returns:
            BM25 score.
        """
        if doc_id not in self.documents:
            return 0.0
        
        doc_tokens = self.documents[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        # Count term frequencies in document
        tf_doc = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in tf_doc:
                continue
            
            tf = tf_doc[term]
            idf = self._idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the index with a query.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Find candidate documents (containing at least one query term)
        candidate_docs: Set[int] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token])
        
        # Score candidates
        scores = []
        for doc_id in candidate_docs:
            score = self._score_document(doc_id, query_tokens)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class FAISSIndex:
    """
    FAISS dense retrieval index with incremental update support.
    
    Uses a simple flat index for exact search. For larger graphs,
    consider using IVF or HNSW indices.
    
    Supports lazy dimension detection: if dimension=0 is passed,
    the dimension is automatically inferred from the first embedding added.
    """
    
    def __init__(self, dimension: int = 0):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension. If 0, auto-detect from first embedding.
        """
        self.dimension = dimension  # 0 means lazy init
        self.embeddings: Dict[int, np.ndarray] = {}
        self._index_built = False
        self._faiss_index = None
        self._id_map: List[int] = []
    
    def add_embedding(self, doc_id: int, embedding: np.ndarray) -> None:
        """
        Add an embedding to the index.
        
        Args:
            doc_id: Document identifier.
            embedding: Embedding vector.
        """
        # Lazy dimension detection
        if self.dimension == 0:
            self.dimension = embedding.shape[0]
            logger.info(f"FAISSIndex: Auto-detected embedding dimension = {self.dimension}")
        
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} != expected {self.dimension}"
            )
        
        self.embeddings[doc_id] = embedding.astype(np.float32)
        self._index_built = False
    
    def remove_embedding(self, doc_id: int) -> None:
        """
        Remove an embedding from the index.
        
        Args:
            doc_id: Document identifier.
        """
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
            self._index_built = False
    
    def _build_index(self) -> None:
        """Build or rebuild the FAISS index with GPU acceleration if available."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, using numpy-based search")
            self._faiss_index = None
            self._index_built = True
            return
        
        if not self.embeddings:
            self._faiss_index = None
            self._index_built = True
            return
        
        # Build ID map and embedding matrix
        self._id_map = list(self.embeddings.keys())
        embedding_matrix = np.stack([self.embeddings[i] for i in self._id_map])
        
        # Create flat index with GPU acceleration if available
        cpu_index = faiss.IndexFlatIP(self.dimension)  # Inner product
        
        # Try to use GPU if available
        try:
            res = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.info("Using GPU-accelerated FAISS index")
        except (RuntimeError, AttributeError):
            # GPU not available or not supported, use CPU
            self._faiss_index = cpu_index
            logger.info("Using CPU-based FAISS index")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        self._faiss_index.add(embedding_matrix)
        
        self._index_built = True
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of (doc_id, similarity) tuples sorted by similarity descending.
        """
        if not self.embeddings:
            return []
        
        if not self._index_built:
            self._build_index()
        
        query = query_embedding.astype(np.float32).reshape(1, -1)
        
        if self._faiss_index is not None:
            try:
                import faiss
                faiss.normalize_L2(query)
                distances, indices = self._faiss_index.search(query, min(top_k, len(self._id_map)))
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self._id_map):
                        results.append((self._id_map[idx], float(distances[0][i])))
                
                return results
            except Exception as e:
                logger.warning(f"FAISS search failed, falling back to numpy: {e}")
        
        # Numpy fallback
        return self._numpy_search(query.flatten(), top_k)
    
    def _numpy_search(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Numpy-based cosine similarity search.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of (doc_id, similarity) tuples.
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        scores = []
        for doc_id, embedding in self.embeddings.items():
            # Normalize document embedding
            doc_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            # Cosine similarity
            similarity = float(np.dot(query_norm, doc_norm))
            scores.append((doc_id, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class UnifiedIndex:
    """
    Unified indexing system combining BM25 and FAISS indices.
    
    Provides a single interface for adding, updating, and searching
    event nodes using both sparse and dense retrieval.
    """
    
    def __init__(self, embedding_dim: Optional[int] = None):
        """
        Initialize unified index.
        
        Args:
            embedding_dim: Dimension of dense embeddings. If None, uses config value.
                           If config value is 0, auto-detects from first embedding.
        """
        self._config = get_pipeline_config()
        
        # Use provided dimension, or config value, or 0 for lazy detection
        if embedding_dim is not None:
            dim = embedding_dim
        else:
            dim = getattr(self._config, 'embedding_dim', 0)
        
        self.bm25 = BM25Index()
        self.faiss = FAISSIndex(dimension=dim)
        
        # Track indexed nodes
        self.indexed_nodes: Set[int] = set()
    
    def add_node(self, node: AdaptiveEventNode) -> None:
        """
        Add an event node to both indices.
        
        Args:
            node: Event node to index.
        """
        # Add to BM25 index
        searchable_text = node.get_searchable_text()
        self.bm25.add_document(node.node_id, searchable_text)
        
        # Add to FAISS index if embedding exists
        embedding = node.get_embedding_array()
        if embedding is not None:
            self.faiss.add_embedding(node.node_id, embedding)
        
        self.indexed_nodes.add(node.node_id)
    
    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from both indices.
        
        Args:
            node_id: ID of node to remove.
        """
        self.bm25.remove_document(node_id)
        self.faiss.remove_embedding(node_id)
        self.indexed_nodes.discard(node_id)
    
    def update_node_embedding(self, node: AdaptiveEventNode) -> None:
        """
        Update the embedding for a node in FAISS index.
        
        Args:
            node: Event node with updated embedding.
        """
        embedding = node.get_embedding_array()
        if embedding is not None:
            self.faiss.add_embedding(node.node_id, embedding)
    
    def search_sparse(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Search using BM25 sparse retrieval.
        
        Args:
            query: Search query.
            top_k: Number of results.
            
        Returns:
            List of (node_id, score) tuples.
        """
        if top_k is None:
            top_k = self._config.anchor_topk_sparse
        return self.bm25.search(query, top_k)
    
    def search_dense(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Search using FAISS dense retrieval.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.
            
        Returns:
            List of (node_id, similarity) tuples.
        """
        if top_k is None:
            top_k = self._config.anchor_topk_dense
        return self.faiss.search(query_embedding, top_k)
    
    def search_hybrid(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        sparse_weight: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Hybrid search combining sparse and dense retrieval using RRF.
        
        Args:
            query: Search query.
            query_embedding: Optional query embedding. If None, will be generated.
            top_k: Number of results.
            sparse_weight: Weight for sparse results in fusion (0-1).
            
        Returns:
            List of (node_id, combined_score) tuples.
        """
        if top_k is None:
            top_k = self._config.anchor_fusion_k
        
        # Get sparse results
        sparse_results = self.search_sparse(query, top_k * 2)
        
        # Get dense results
        if query_embedding is None:
            try:
                embeddings = get_embedding_with_retry(
                    self._config.embedding_model,
                    query
                )
                query_embedding = np.array(embeddings[0], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to get query embedding: {e}")
                return sparse_results[:top_k]
        
        dense_results = self.search_dense(query_embedding, top_k * 2)
        
        # Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}
        k = 60  # RRF constant
        
        for rank, (doc_id, _) in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)
        
        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - sparse_weight) / (k + rank + 1)
        
        # Sort by combined score
        combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return combined[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics.
        """
        return {
            "num_indexed_nodes": len(self.indexed_nodes),
            "bm25_total_docs": self.bm25.total_docs,
            "bm25_vocab_size": len(self.bm25.doc_frequencies),
            "faiss_num_embeddings": len(self.faiss.embeddings),
        }


def generate_node_embeddings(
    nodes: List[AdaptiveEventNode],
    model_name: Optional[str] = None,
    batch_size: int = 20
) -> None:
    """
    Generate embeddings for a list of nodes.
    
    Modifies nodes in place by setting their text_embedding field.
    
    Args:
        nodes: List of event nodes.
        model_name: Embedding model to use.
        batch_size: Number of texts to embed at once.
    """
    config = get_pipeline_config()
    
    if model_name is None:
        model_name = config.embedding_model
    
    # Filter nodes without embeddings
    nodes_to_embed = [n for n in nodes if n.text_embedding is None]
    
    if not nodes_to_embed:
        return
    
    logger.info(f"Generating embeddings for {len(nodes_to_embed)} nodes")
    
    for i in range(0, len(nodes_to_embed), batch_size):
        batch = nodes_to_embed[i:i + batch_size]
        texts = [node.summary_text or node.get_searchable_text() for node in batch]
        
        try:
            embeddings = get_embedding_with_retry(model_name, texts)
            
            for node, embedding in zip(batch, embeddings):
                node.text_embedding = embedding
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {i}: {e}")

