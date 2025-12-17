#!/usr/bin/env python3
"""
Main entry point for AMR-Graph pipeline.

This script provides CLI interface for:
- Building memory graphs from videos
- Running QA evaluation on M3-Bench
- Testing retrieval on custom queries
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import (
    get_pipeline_config,
    set_pipeline_config,
    PipelineConfig,
    PROJECT_ROOT
)
from event_node import (
    AdaptiveEventNode,
    EventNodeFactory,
    get_node_factory,
    reset_node_factory
)
from amr_graph import AMRGraph
from video_processor import VideoProcessor, process_video_to_nodes
from indexing import UnifiedIndex, generate_node_embeddings
from retrieval import RetrievalPipeline
from evaluation import (
    AnswerEvaluator,
    ResultTracker,
    load_qa_data,
    format_results_for_comparison
)


logger = logging.getLogger(__name__)


def print_results_table(tracker: ResultTracker, save_path: Optional[str] = None) -> None:
    """
    Print evaluation results in a formatted table.
    
    Shows Overall Accuracy and accuracy by question type as per rule.md item 15.
    
    Args:
        tracker: ResultTracker containing evaluation results.
        save_path: Optional path to save the table to file.
    """
    accuracy = tracker.get_accuracy()
    
    # Define the 5 question types from M3-Bench
    question_types = [
        "Person Understanding",
        "Cross-Modal Reasoning", 
        "Multi-Hop Reasoning",
        "Multi-Detail Reasoning",
        "Temporal Reasoning"
    ]
    
    # Get counts by type (handles multiple types per question)
    type_counts = tracker.get_counts_by_type()
    
    # Build table lines
    table_lines = []
    table_lines.append("=" * 70)
    table_lines.append("EVALUATION RESULTS TABLE")
    table_lines.append("=" * 70)
    table_lines.append(f"{'Question Type':<30} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<10}")
    table_lines.append("-" * 70)
    
    # Add each standard M3-Bench type
    for q_type in question_types:
        if q_type in type_counts:
            counts = type_counts[q_type]
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            table_lines.append(f"{q_type:<30} | {counts['correct']:<10} | {counts['total']:<10} | {acc:.2%}")
    
    # Add any other types not in the standard list
    for q_type, counts in sorted(type_counts.items()):
        if q_type not in question_types:
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            table_lines.append(f"{q_type:<30} | {counts['correct']:<10} | {counts['total']:<10} | {acc:.2%}")
    
    table_lines.append("-" * 70)
    
    # Add overall (direct count of correct questions, not aggregated by type)
    total_correct = sum(1 for r in tracker.results if r["is_correct"])
    total = len(tracker.results)
    table_lines.append(f"{'OVERALL':<30} | {total_correct:<10} | {total:<10} | {accuracy:.2%}")
    table_lines.append("=" * 70)
    
    # Add timing summary
    table_lines.append("")
    table_lines.append("TIMING BREAKDOWN:")
    table_lines.append("-" * 50)
    total_time = sum(tracker.timing.values())
    for stage, duration in sorted(tracker.timing.items(), key=lambda x: -x[1]):
        pct = (duration / total_time * 100) if total_time > 0 else 0
        table_lines.append(f"  {stage:<35}: {duration:8.2f}s ({pct:5.1f}%)")
    table_lines.append("-" * 50)
    table_lines.append(f"  {'TOTAL TIME':<35}: {total_time:8.2f}s")
    table_lines.append("=" * 70)
    
    # Print to console
    for line in table_lines:
        print(line)
    
    # Save to file if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(table_lines))
        logger.info(f"Saved results table to {save_path}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def build_memory_graph(
    video_path: str,
    output_path: Optional[str] = None,
    max_clips: Optional[int] = None,
    target_clip_sec: Optional[float] = None
) -> AMRGraph:
    """
    Build memory graph from a video file.
    
    Args:
        video_path: Path to video file.
        output_path: Optional path to save the graph.
        max_clips: Optional maximum number of clips to process.
        target_clip_sec: Target clip duration in seconds.
        
    Returns:
        Built AMRGraph instance.
    """
    video_id = Path(video_path).stem
    
    logger.info(f"Building memory graph for: {video_path}")
    logger.info(f"Video ID: {video_id}")
    
    # Reset node factory for fresh graph
    reset_node_factory()
    factory = get_node_factory()
    
    # Initialize graph and index
    graph = AMRGraph(video_id=video_id)
    index = UnifiedIndex()
    
    # Initialize video processor
    processor = VideoProcessor(video_path, node_factory=factory)
    
    try:
        clip_count = 0
        
        for nodes in processor.process_all_clips(
            target_clip_sec=target_clip_sec,
            max_clips=max_clips
        ):
            # Add nodes to graph (this creates edges)
            for node in nodes:
                graph.process_new_node(node)
            
            # Generate embeddings for batch
            generate_node_embeddings(nodes)
            
            # Add to index
            for node in nodes:
                index.add_node(node)
            
            clip_count += 1
            logger.info(f"Processed clip {clip_count}, total nodes: {graph.num_nodes}")
        
    finally:
        processor.cleanup()
    
    # Save graph if output path specified
    if output_path:
        graph.save_to_json(output_path)
        logger.info(f"Saved memory graph to: {output_path}")
    
    logger.info(f"Graph statistics: {graph.get_statistics()}")
    
    return graph


def run_qa_evaluation(
    graph: AMRGraph,
    qa_data: List[Dict[str, Any]],
    output_dir: str,
    video_name: str,
    tracker: Optional[ResultTracker] = None
) -> ResultTracker:
    """
    Run QA evaluation on a memory graph.
    
    Args:
        graph: Built memory graph.
        qa_data: List of QA items to evaluate.
        output_dir: Directory to save results.
        video_name: Name of the video being evaluated.
        tracker: Optional existing ResultTracker to continue using.
        
    Returns:
        ResultTracker with all results.
    """
    config = get_pipeline_config()
    
    # Use existing tracker or create new one
    if tracker is None:
        tracker = ResultTracker()
    
    # Build index from graph
    logger.info("Building search index from graph...")
    tracker.start_stage("index_building")
    index = UnifiedIndex()
    
    for node_id, node in graph.nodes.items():
        # Generate embedding if missing
        if node.text_embedding is None:
            generate_node_embeddings([node])
        index.add_node(node)
    
    tracker.end_stage("index_building")
    logger.info(f"Index statistics: {index.get_statistics()}")
    
    # Initialize pipeline
    pipeline = RetrievalPipeline(graph, index)
    evaluator = AnswerEvaluator()
    
    # Process each question
    tracker.start_stage("total_qa_evaluation")
    
    for i, qa_item in enumerate(qa_data):
        question_id = qa_item["question_id"]
        question = qa_item["question"]
        ground_truth = qa_item["answer"]
        question_type = qa_item.get("question_type", [])
        
        logger.info(f"\n[{i+1}/{len(qa_data)}] Processing: {question_id}")
        logger.info(f"Question: {question}")
        
        try:
            # Run retrieval and get answer
            tracker.start_stage("retrieval_per_question")
            result = pipeline.retrieve_and_answer(question)
            tracker.end_stage("retrieval_per_question")
            
            predicted = result.final_answer
            
            # Evaluate answer
            tracker.start_stage("evaluation_per_question")
            is_correct, eval_response = evaluator.evaluate(
                question, ground_truth, predicted
            )
            tracker.end_stage("evaluation_per_question")
            
            logger.info(f"Predicted: {predicted}")
            logger.info(f"Ground Truth: {ground_truth}")
            logger.info(f"Correct: {is_correct}")
            
            # Record result
            tracker.add_result(
                question_id=question_id,
                question=question,
                ground_truth=ground_truth,
                predicted=predicted,
                is_correct=is_correct,
                metadata={
                    "question_types": question_type if question_type else ["Unknown"],
                    "anchor_count": len(result.anchor_nodes),
                    "navigated_count": len(result.navigated_nodes),
                    "verified_count": len(result.verified_nodes),
                    "context": result.context_text[:500]
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process question {question_id}: {e}")
            tracker.add_result(
                question_id=question_id,
                question=question,
                ground_truth=ground_truth,
                predicted="ERROR: " + str(e),
                is_correct=False,
                metadata={"error": str(e)}
            )
    
    tracker.end_stage("total_qa_evaluation")
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results_path = os.path.join(output_dir, f"{video_name}_results.jsonl")
    tracker.save_results(results_path)
    
    summary_path = os.path.join(output_dir, f"{video_name}_summary.json")
    tracker.save_summary(summary_path)
    
    timing_path = os.path.join(output_dir, f"{video_name}_timing.txt")
    tracker.save_timing_report(timing_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Video: {video_name}")
    print(f"Total Questions: {len(qa_data)}")
    print(f"Accuracy: {tracker.get_accuracy():.2%}")
    print("\nAccuracy by Type:")
    for q_type, acc in tracker.get_accuracy_by_type().items():
        print(f"  {q_type}: {acc:.2%}")
    print("=" * 60)
    
    return tracker


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AMR-Graph: Adaptive Multi-Channel Relational Memory Graph Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build memory graph from video")
    build_parser.add_argument("--video_path", help="Path to video file")
    build_parser.add_argument("--output", "-o", help="Output path for graph JSON")
    build_parser.add_argument("--max-clips", type=int, help="Maximum clips to process")
    build_parser.add_argument("--clip-duration", type=float, default=None,
                             help="Target clip duration in seconds (default: from config)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run QA evaluation")
    eval_parser.add_argument("--video_path", help="Path to video file")
    eval_parser.add_argument("--annotations", "-a", 
                            default=str(PROJECT_ROOT / "data" / "annotations" / "robot.json"),
                            help="Path to annotations file")
    eval_parser.add_argument("--video-name", help="Video name to filter QA (default: from filename)")
    eval_parser.add_argument("--output-dir", "-o",
                            default=str(PROJECT_ROOT / "data" / "results"),
                            help="Output directory for results")
    eval_parser.add_argument("--max-clips", type=int, help="Maximum clips to process")
    eval_parser.add_argument("--clip-duration", type=float, default=None,
                             help="Target clip duration in seconds (default: from config)")
    eval_parser.add_argument("--graph-path", help="Load existing graph instead of building")
    
    # Query command for testing
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("graph_path", help="Path to saved graph JSON")
    query_parser.add_argument("question", help="Question to answer")
    
    # Common arguments
    for p in [build_parser, eval_parser, query_parser]:
        p.add_argument("--log-level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        p.add_argument("--log-file", help="Path to log file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.log_level, getattr(args, 'log_file', None))
    
    # Execute command
    if args.command == "build":
        output_path = args.output
        if not output_path:
            video_name = Path(args.video_path).stem
            output_path = str(PROJECT_ROOT / "data" / "memory_graphs" / f"{video_name}_graph.json")
        
        graph = build_memory_graph(
            video_path=args.video_path,
            output_path=output_path,
            max_clips=args.max_clips,
            target_clip_sec=args.clip_duration
        )
        
        print(f"\nGraph built successfully!")
        print(f"Saved to: {output_path}")
        print(f"Statistics: {graph.get_statistics()}")
    
    elif args.command == "evaluate":
        # Determine video name
        video_name = args.video_name or Path(args.video_path).stem
        
        # Initialize tracker for all stages
        main_tracker = ResultTracker()
        
        # Load QA data
        logger.info(f"Loading QA data from: {args.annotations}")
        qa_data = load_qa_data(args.annotations, video_name)
        
        if not qa_data:
            logger.error(f"No QA data found for video: {video_name}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(qa_data)} QA items for {video_name}")
        
        # Build or load graph
        if args.graph_path:
            logger.info(f"Loading existing graph from: {args.graph_path}")
            graph = AMRGraph.load_from_json(args.graph_path)
        else:
            # Build graph - track both video preprocessing and graph building
            graph_path = str(PROJECT_ROOT / "data" / "memory_graphs" / f"{video_name}_graph.json")
            
            logger.info("Building memory graph from video...")
            main_tracker.start_stage("memory_graph_building")
            
            graph = build_memory_graph(
                video_path=args.video_path,
                output_path=graph_path,
                max_clips=args.max_clips,
                target_clip_sec=args.clip_duration
            )
            
            main_tracker.end_stage("memory_graph_building")
            logger.info(f"Memory graph building took {main_tracker.timing['memory_graph_building']:.2f}s")
        
        # Run evaluation (pass tracker to continue timing)
        result_tracker = run_qa_evaluation(
            graph=graph,
            qa_data=qa_data,
            output_dir=args.output_dir,
            video_name=video_name,
            tracker=main_tracker
        )
        
        # Print detailed results
        print("\n" + format_results_for_comparison(result_tracker.results))
        
        # Print and save results table 
        table_path = os.path.join(args.output_dir, f"{video_name}_acctable.txt")
        print_results_table(result_tracker, save_path=table_path)
    
    elif args.command == "query":
        # Load graph
        logger.info(f"Loading graph from: {args.graph_path}")
        graph = AMRGraph.load_from_json(args.graph_path)
        
        # Build index
        logger.info("Building search index...")
        index = UnifiedIndex()
        
        nodes_without_embedding = []
        for node_id, node in graph.nodes.items():
            if node.text_embedding is None:
                nodes_without_embedding.append(node)
            index.add_node(node)
        
        if nodes_without_embedding:
            logger.info(f"Generating embeddings for {len(nodes_without_embedding)} nodes...")
            generate_node_embeddings(nodes_without_embedding)
            for node in nodes_without_embedding:
                index.update_node_embedding(node)
        
        # Run query
        pipeline = RetrievalPipeline(graph, index)
        
        logger.info(f"Query: {args.question}")
        result = pipeline.retrieve_and_answer(args.question)
        
        print("\n" + "=" * 60)
        print("QUERY RESULT")
        print("=" * 60)
        print(f"Question: {args.question}")
        print(f"\nAnswer: {result.final_answer}")
        print(f"\nAnchors: {len(result.anchor_nodes)} nodes")
        print(f"Navigated: {len(result.navigated_nodes)} nodes")
        print(f"Verified: {len(result.verified_nodes)} nodes")
        print(f"\nContext: {result.context_text[:500]}...")
        print("=" * 60)


if __name__ == "__main__":
    main()
