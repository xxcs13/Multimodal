"""
Evaluation module for AMR-Graph pipeline.

This module provides evaluation utilities aligned with M3-Bench,
including answer verification using GPT-4o and result tracking.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from config import get_pipeline_config, PROJECT_ROOT
from api_utils import call_llm_with_retry, build_text_message
from prompts import PROMPT_VERIFY_ANSWER, PROMPT_VERIFY_ANSWER_INFERENCE


logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """
    Evaluates predicted answers against ground truth using LLM verification.
    
    This aligns with M3-Bench evaluation methodology.
    """
    
    def __init__(self, model_name: Optional[str] = None, use_inference: bool = True):
        """
        Initialize evaluator.
        
        Args:
            model_name: LLM model for evaluation.
            use_inference: Whether to use inference-based evaluation (recommended).
        """
        config = get_pipeline_config()
        self.model_name = model_name or "openai/gpt-4o"
        self.use_inference = use_inference
    
    def evaluate(
        self,
        question: str,
        ground_truth: str,
        predicted: str
    ) -> Tuple[bool, str]:
        """
        Evaluate if predicted answer is correct.
        
        Args:
            question: The question asked.
            ground_truth: Ground truth answer.
            predicted: Model's predicted answer.
            
        Returns:
            Tuple of (is_correct, raw_response).
        """
        if not predicted or not predicted.strip():
            return False, "Empty prediction"
        
        # Select prompt
        if self.use_inference:
            prompt = PROMPT_VERIFY_ANSWER_INFERENCE.format(
                question=question,
                ground_truth=ground_truth,
                predicted=predicted
            )
        else:
            prompt = PROMPT_VERIFY_ANSWER.format(
                question=question,
                ground_truth=ground_truth,
                predicted=predicted
            )
        
        try:
            response, _ = call_llm_with_retry(
                model_name=self.model_name,
                messages=[build_text_message(prompt)],
                temperature=0.0,
                max_tokens=50
            )
            
            response_lower = response.lower().strip()
            is_correct = "yes" in response_lower
            
            return is_correct, response
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False, str(e)


class ResultTracker:
    """
    Tracks evaluation results and timing information.
    """
    
    def __init__(self):
        """Initialize result tracker."""
        self.results: List[Dict[str, Any]] = []
        self.timing: Dict[str, float] = {}
        self._stage_start_times: Dict[str, float] = {}
    
    def start_stage(self, stage_name: str) -> None:
        """
        Start timing a stage.
        
        Args:
            stage_name: Name of the stage.
        """
        self._stage_start_times[stage_name] = time.time()
    
    def end_stage(self, stage_name: str) -> float:
        """
        End timing a stage and record duration.
        
        Args:
            stage_name: Name of the stage.
            
        Returns:
            Duration in seconds.
        """
        if stage_name not in self._stage_start_times:
            return 0.0
        
        duration = time.time() - self._stage_start_times[stage_name]
        
        if stage_name not in self.timing:
            self.timing[stage_name] = 0.0
        self.timing[stage_name] += duration
        
        del self._stage_start_times[stage_name]
        
        return duration
    
    def add_result(
        self,
        question_id: str,
        question: str,
        ground_truth: str,
        predicted: str,
        is_correct: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an evaluation result.
        
        Args:
            question_id: Unique question identifier.
            question: The question text.
            ground_truth: Ground truth answer.
            predicted: Predicted answer.
            is_correct: Whether the prediction is correct.
            metadata: Optional additional metadata.
        """
        result = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            result["metadata"] = metadata
        
        self.results.append(result)
    
    def get_accuracy(self) -> float:
        """
        Calculate overall accuracy.
        
        Returns:
            Accuracy as a float between 0 and 1.
        """
        if not self.results:
            return 0.0
        
        correct = sum(1 for r in self.results if r["is_correct"])
        return correct / len(self.results)
    
    def get_accuracy_by_type(self) -> Dict[str, float]:
        """
        Calculate accuracy by question type.
        
        Returns:
            Dictionary mapping question type to accuracy.
        """
        type_results: Dict[str, List[bool]] = {}
        
        for result in self.results:
            q_type = result.get("metadata", {}).get("question_type", "Unknown")
            if q_type not in type_results:
                type_results[q_type] = []
            type_results[q_type].append(result["is_correct"])
        
        return {
            q_type: sum(results) / len(results) if results else 0.0
            for q_type, results in type_results.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all results.
        
        Returns:
            Summary dictionary.
        """
        return {
            "total_questions": len(self.results),
            "correct": sum(1 for r in self.results if r["is_correct"]),
            "accuracy": self.get_accuracy(),
            "accuracy_by_type": self.get_accuracy_by_type(),
            "timing": self.timing,
            "total_time": sum(self.timing.values())
        }
    
    def save_results(self, filepath: str) -> None:
        """
        Save results to JSONL file.
        
        Args:
            filepath: Path to save results.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.results)} results to {filepath}")
    
    def save_summary(self, filepath: str) -> None:
        """
        Save summary to JSON file.
        
        Args:
            filepath: Path to save summary.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_summary()
        summary["results"] = self.results
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary to {filepath}")
    
    def save_timing_report(self, filepath: str) -> None:
        """
        Save detailed timing report.
        
        Args:
            filepath: Path to save timing report.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        total_time = sum(self.timing.values())
        
        report_lines = [
            "=" * 60,
            "TIMING REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "Stage Breakdown:",
            "-" * 40
        ]
        
        for stage, duration in sorted(self.timing.items(), key=lambda x: -x[1]):
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            report_lines.append(f"  {stage:30s}: {duration:8.2f}s ({percentage:5.1f}%)")
        
        report_lines.extend([
            "-" * 40,
            f"  {'TOTAL':30s}: {total_time:8.2f}s",
            "",
            "Performance Metrics:",
            "-" * 40,
            f"  Questions processed: {len(self.results)}",
            f"  Accuracy: {self.get_accuracy():.2%}",
            f"  Avg time per question: {total_time / max(len(self.results), 1):.2f}s",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Saved timing report to {filepath}")
        
        # Also print to console
        print(report)


def load_qa_data(annotation_path: str, video_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load QA data from M3-Bench annotation file.
    
    Args:
        annotation_path: Path to annotation JSON file.
        video_name: Optional video name to filter by.
        
    Returns:
        List of QA items with video and question info.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    qa_list = []
    
    for vid_name, vid_data in data.items():
        if video_name and vid_name != video_name:
            continue
        
        video_path = vid_data.get("video_path", "")
        mem_path = vid_data.get("mem_path", "")
        
        for qa in vid_data.get("qa_list", []):
            qa_item = {
                "video_name": vid_name,
                "video_path": video_path,
                "mem_path": mem_path,
                "question_id": qa.get("question_id", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "question_type": qa.get("type", []),
                "timestamp": qa.get("timestamp", ""),
                "before_clip": qa.get("before_clip"),
                "reasoning": qa.get("reasoning", "")
            }
            qa_list.append(qa_item)
    
    return qa_list


def format_results_for_comparison(
    results: List[Dict[str, Any]]
) -> str:
    """
    Format results for easy comparison and review.
    
    Args:
        results: List of result dictionaries.
        
    Returns:
        Formatted string for display.
    """
    lines = []
    
    for i, result in enumerate(results, 1):
        status = "Correct" if result["is_correct"] else "Incorrect"
        lines.append(f"\n{'='*60}")
        lines.append(f"Question {i}: {result['question_id']}")
        lines.append(f"Status: {status}")
        lines.append(f"Q: {result['question']}")
        lines.append(f"Ground Truth: {result['ground_truth']}")
        lines.append(f"Predicted: {result['predicted']}")
        
        if "metadata" in result:
            meta = result["metadata"]
            if "question_type" in meta:
                lines.append(f"Type: {meta['question_type']}")
    
    lines.append(f"\n{'='*60}")
    
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    lines.append(f"Overall: {correct}/{total} ({correct/total*100:.1f}%)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test evaluation module
    logging.basicConfig(level=logging.INFO)
    
    # Test evaluator
    print("Testing Answer Evaluator:")
    evaluator = AnswerEvaluator()
    
    test_cases = [
        {
            "question": "What kind of coffee does Lily like?",
            "ground_truth": "Mocha Coffee",
            "predicted": "Lily likes mocha coffee"
        },
        {
            "question": "What is Emma's major?",
            "ground_truth": "Computer ; Programming",
            "predicted": "Emma studies computer science"
        },
        {
            "question": "Is Emma's tidying up habit good?",
            "ground_truth": "Not good",
            "predicted": "Yes, she is very organized"
        }
    ]
    
    for tc in test_cases:
        is_correct, response = evaluator.evaluate(
            tc["question"],
            tc["ground_truth"],
            tc["predicted"]
        )
        print(f"\nQuestion: {tc['question']}")
        print(f"Ground Truth: {tc['ground_truth']}")
        print(f"Predicted: {tc['predicted']}")
        print(f"Correct: {is_correct} (Response: {response})")
    
    # Test result tracker
    print("\n\nTesting Result Tracker:")
    tracker = ResultTracker()
    
    tracker.start_stage("video_processing")
    time.sleep(0.1)  # Simulate work
    tracker.end_stage("video_processing")
    
    tracker.start_stage("retrieval")
    time.sleep(0.05)
    tracker.end_stage("retrieval")
    
    for i, tc in enumerate(test_cases):
        is_correct = i == 0  # Just for testing
        tracker.add_result(
            question_id=f"Q{i+1}",
            question=tc["question"],
            ground_truth=tc["ground_truth"],
            predicted=tc["predicted"],
            is_correct=is_correct,
            metadata={"question_type": "Person_Understanding"}
        )
    
    print(f"\nSummary: {tracker.get_summary()}")
    
    # Test QA loading
    print("\n\nTesting QA Data Loading:")
    annotation_path = PROJECT_ROOT / "data" / "annotations" / "robot.json"
    if annotation_path.exists():
        qa_data = load_qa_data(str(annotation_path), "bedroom_01")
        print(f"Loaded {len(qa_data)} QA items for bedroom_01")
        if qa_data:
            print(f"First question: {qa_data[0]['question']}")
