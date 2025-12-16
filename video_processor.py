"""
Video processing module for AMR-Graph.

This module handles video clip segmentation and multimodal analysis
using LLM (Gemini via OpenRouter) to extract structured event information.
"""

import os
import json
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Tuple

from config import get_pipeline_config
from api_utils import (
    call_llm_with_retry,
    build_multimodal_message,
    parse_json_response,
    video_to_base64
)
from prompts import (
    PROMPT_CLIP_ANALYSIS,
    PROMPT_CLIP_ANALYSIS_STRICT_JSON,
    PROMPT_JSON_REPAIR,
    PROMPT_SEMANTIC_DESCRIPTION,
    PROMPT_NEIGHBOR_INTERPOLATION
)
from event_node import AdaptiveEventNode, EventNodeFactory, get_node_factory


logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get video duration: {result.stderr}")
    
    return float(result.stdout.strip())


def extract_clip(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float
) -> str:
    """
    Extract a clip from a video file.
    
    Args:
        video_path: Path to source video.
        output_path: Path for output clip.
        start_time: Start time in seconds.
        duration: Duration in seconds.
        
    Returns:
        Path to the extracted clip.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-loglevel", "error",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract clip: {result.stderr}")
    
    return output_path


def generate_clip_segments(
    video_path: str,
    target_clip_sec: Optional[float] = None
) -> Generator[Tuple[str, float, float], None, None]:
    """
    Generate clip segment information for a video.
    
    Yields tuples of (clip_id, start_time, end_time) without extracting clips.
    
    Args:
        video_path: Path to the video file.
        target_clip_sec: Target clip length in seconds.
        
    Yields:
        Tuples of (clip_id, start_time, end_time).
    """
    config = get_pipeline_config()
    
    if target_clip_sec is None:
        target_clip_sec = config.target_clip_sec
    
    video_duration = get_video_duration(video_path)
    video_id = Path(video_path).stem
    
    t = 0.0
    clip_idx = 0
    
    while t < video_duration:
        end_time = min(t + target_clip_sec, video_duration)
        clip_id = f"{video_id}_clip_{clip_idx:04d}"
        
        yield (clip_id, t, end_time)
        
        t = end_time
        clip_idx += 1


class ClipAnalyzer:
    """
    Analyzes video clips using multimodal LLM to extract structured events.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the clip analyzer.
        
        Args:
            model_name: LLM model to use for analysis.
        """
        config = get_pipeline_config()
        self.model_name = model_name or config.llm_model_clip_analysis
        self._temp_dir = None
    
    def analyze_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        clip_id: str,
        prev_summary: Optional[str] = None,
        next_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a video clip and extract structured event information.
        
        Args:
            video_path: Path to the full video.
            start_time: Clip start time in seconds.
            end_time: Clip end time in seconds.
            clip_id: Identifier for this clip.
            prev_summary: Summary from previous clip for fallback interpolation.
            next_summary: Summary from next clip for fallback interpolation.
            
        Returns:
            Dictionary containing parsed clip analysis.
        """
        # Extract clip to temporary file
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp()
        
        clip_path = os.path.join(self._temp_dir, f"{clip_id}.mp4")
        duration = end_time - start_time
        
        try:
            extract_clip(video_path, clip_path, start_time, duration)
            
            # Convert clip to base64
            clip_base64 = video_to_base64(clip_path)
            
            # Build message with video and prompt
            message = build_multimodal_message(
                text=PROMPT_CLIP_ANALYSIS,
                video_base64=clip_base64
            )
            
            # Call LLM
            response, tokens = call_llm_with_retry(
                model_name=self.model_name,
                messages=[message],
                temperature=0.1,
                max_tokens=4096
            )
            
            logger.debug(f"Clip {clip_id} analysis used {tokens} tokens")
            
            # Parse JSON response
            try:
                result = parse_json_response(response)
            except json.JSONDecodeError as e:
                # Retry with stricter prompt
                logger.warning(f"JSON parse failed for {clip_id}, retrying with strict prompt")
                
                message = build_multimodal_message(
                    text=PROMPT_CLIP_ANALYSIS_STRICT_JSON,
                    video_base64=clip_base64
                )
                
                response, _ = call_llm_with_retry(
                    model_name=self.model_name,
                    messages=[message],
                    temperature=0.0,
                    max_tokens=4096
                )
                
                try:
                    result = parse_json_response(response)
                except json.JSONDecodeError as e2:
                    # Use multi-level fallback strategy
                    logger.warning(f"JSON parse still failed for {clip_id}, using multi-level fallback")
                    result = self._create_fallback_result(
                        response=response,
                        clip_id=clip_id,
                        start_time=start_time,
                        end_time=end_time,
                        video_path=video_path,
                        prev_summary=prev_summary,
                        next_summary=next_summary
                    )
            
            # Add metadata
            result["clip_id"] = clip_id
            result["clip_start_time"] = start_time
            result["clip_end_time"] = end_time
            
            return result
            
        finally:
            # Cleanup temp clip
            if os.path.exists(clip_path):
                os.remove(clip_path)
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
    
    def _try_json_repair(
        self,
        broken_json: str,
        clip_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Level 1 Fallback: Attempt to repair broken JSON using LLM.
        
        Args:
            broken_json: The malformed JSON response from clip analysis.
            clip_id: Identifier for this clip.
            
        Returns:
            Repaired JSON dict if successful, None otherwise.
        """
        config = get_pipeline_config()
        
        try:
            prompt = PROMPT_JSON_REPAIR.format(broken_json=broken_json[:4000])
            
            message = {"role": "user", "content": prompt}
            
            response, _ = call_llm_with_retry(
                model_name=config.llm_model_summarization,
                messages=[message],
                temperature=0.0,
                max_tokens=4096,
                retry_count=1
            )
            
            result = parse_json_response(response)
            logger.info(f"JSON repair successful for {clip_id}")
            return result
            
        except Exception as e:
            logger.warning(f"JSON repair failed for {clip_id}: {e}")
            return None
    
    def _try_semantic_description(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        clip_id: str
    ) -> Optional[str]:
        """
        Level 2 Fallback: Get semantic description without JSON structure.
        
        Makes a simpler multimodal LLM call that only returns natural language
        description, avoiding JSON parsing issues entirely.
        
        Args:
            video_path: Path to the full video.
            start_time: Clip start time in seconds.
            end_time: Clip end time in seconds.
            clip_id: Identifier for this clip.
            
        Returns:
            Semantic description string if successful, None otherwise.
        """
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp()
        
        clip_path = os.path.join(self._temp_dir, f"{clip_id}_fallback.mp4")
        duration = end_time - start_time
        
        try:
            extract_clip(video_path, clip_path, start_time, duration)
            clip_base64 = video_to_base64(clip_path)
            
            message = build_multimodal_message(
                text=PROMPT_SEMANTIC_DESCRIPTION,
                video_base64=clip_base64
            )
            
            response, _ = call_llm_with_retry(
                model_name=self.model_name,
                messages=[message],
                temperature=0.1,
                max_tokens=4096,
                retry_count=2
            )
            
            description = response.strip()
            
            if description and len(description.split()) >= 5:
                logger.info(f"Semantic description successful for {clip_id}")
                return description
            
            return None
            
        except Exception as e:
            logger.warning(f"Semantic description failed for {clip_id}: {e}")
            return None
            
        finally:
            if os.path.exists(clip_path):
                os.remove(clip_path)
    
    def _try_neighbor_interpolation(
        self,
        start_time: float,
        end_time: float,
        clip_id: str,
        prev_summary: Optional[str] = None,
        next_summary: Optional[str] = None
    ) -> Optional[str]:
        """
        Level 3 Fallback: Interpolate from neighboring clip summaries.
        
        Uses LLM to generate a reasonable description based on what happened
        before and after this clip.
        
        Args:
            start_time: Clip start time in seconds.
            end_time: Clip end time in seconds.
            clip_id: Identifier for this clip.
            prev_summary: Summary text from the previous clip (if available).
            next_summary: Summary text from the next clip (if available).
            
        Returns:
            Interpolated description string if successful, None otherwise.
        """
        config = get_pipeline_config()
        
        if prev_summary is None and next_summary is None:
            return None
        
        neighbor_context_parts = []
        if prev_summary:
            neighbor_context_parts.append(f"Before this clip: {prev_summary}")
        if next_summary:
            neighbor_context_parts.append(f"After this clip: {next_summary}")
        
        neighbor_context = "\n".join(neighbor_context_parts)
        
        try:
            prompt = PROMPT_NEIGHBOR_INTERPOLATION.format(
                time_start=start_time,
                time_end=end_time,
                neighbor_context=neighbor_context
            )
            
            message = {"role": "user", "content": prompt}
            
            response, _ = call_llm_with_retry(
                model_name=config.llm_model_summarization,
                messages=[message],
                temperature=0.3,
                max_tokens=4096,
                retry_count=1
            )
            
            description = response.strip()
            
            if description and len(description.split()) >= 5:
                logger.info(f"Neighbor interpolation successful for {clip_id}")
                return description
            
            return None
            
        except Exception as e:
            logger.warning(f"Neighbor interpolation failed for {clip_id}: {e}")
            return None
    
    def _create_fallback_result(
        self,
        response: str,
        clip_id: str,
        start_time: float,
        end_time: float,
        video_path: Optional[str] = None,
        prev_summary: Optional[str] = None,
        next_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a fallback result using multi-level recovery strategy.
        
        Tries multiple levels of fallback in order:
        1. JSON repair via LLM
        2. Semantic description via multimodal LLM (no JSON)
        3. Neighbor interpolation from adjacent clips
        4. Regex extraction from raw response
        5. Generic placeholder with time information
        
        Args:
            response: Raw LLM response that failed to parse.
            clip_id: Identifier for this clip.
            start_time: Clip start time.
            end_time: Clip end time.
            video_path: Path to video file for Level 2 fallback.
            prev_summary: Summary from previous clip for Level 3 fallback.
            next_summary: Summary from next clip for Level 3 fallback.
            
        Returns:
            Result dictionary with extracted/generated information.
        """
        import re
        
        # Level 1: Try JSON repair
        repaired = self._try_json_repair(response, clip_id)
        if repaired is not None:
            repaired["_fallback_level"] = 1
            return repaired
        
        # Level 2: Try semantic description (requires video path)
        semantic_desc = None
        if video_path:
            semantic_desc = self._try_semantic_description(
                video_path, start_time, end_time, clip_id
            )
        
        if semantic_desc:
            return self._build_result_from_description(
                semantic_desc, start_time, end_time, fallback_level=2
            )
        
        # Level 3: Try neighbor interpolation
        interpolated_desc = self._try_neighbor_interpolation(
            start_time, end_time, clip_id, prev_summary, next_summary
        )
        
        if interpolated_desc:
            return self._build_result_from_description(
                interpolated_desc, start_time, end_time, fallback_level=3
            )
        
        # Level 4: Regex extraction from raw response
        summary = self._extract_summary_from_response(response)
        dialogues = self._extract_dialogues_from_response(response)
        characters = self._extract_characters_from_response(response)
        
        if summary and len(summary.split()) >= 5:
            return self._build_result_from_description(
                summary, start_time, end_time,
                dialogues=dialogues, characters=characters, fallback_level=4
            )
        
        # Level 5: Generic placeholder
        logger.warning(f"All fallback levels exhausted for {clip_id}, using generic placeholder")
        
        generic_summary = (
            f"A video segment from {start_time:.1f}s to {end_time:.1f}s showing "
            "ongoing activity. The scene content could not be fully parsed but "
            "may contain relevant actions, dialogue, or spatial information."
        )
        
        return self._build_result_from_description(
            generic_summary, start_time, end_time,
            dialogues=dialogues, characters=characters, fallback_level=5
        )
    
    def _build_result_from_description(
        self,
        description: str,
        start_time: float,
        end_time: float,
        dialogues: Optional[List[str]] = None,
        characters: Optional[List[str]] = None,
        fallback_level: int = 0
    ) -> Dict[str, Any]:
        """
        Build a structured result dictionary from a text description.
        
        Args:
            description: The semantic description of the clip.
            start_time: Clip start time.
            end_time: Clip end time.
            dialogues: Optional list of extracted dialogue snippets.
            characters: Optional list of character descriptions.
            fallback_level: Which fallback level produced this result.
            
        Returns:
            Structured result dictionary.
        """
        dialogues = dialogues or []
        characters = characters or []
        
        return {
            "clip_summary": description,
            "events": [{
                "summary": description,
                "time_within_clip_start": 0.0,
                "time_within_clip_end": end_time - start_time,
                "characters_involved": characters[:3] if characters else [],
                "objects_mentioned": [],
                "key_actions": [],
                # "local_event_id": "E1",
                # "summary": description,
                # "time_start": 0.0,
                # "time_end": end_time - start_time,
                # "actors": characters[:3] if characters else [],
                # "objects": [],
                "dialogue": dialogues[:5] if dialogues else [],
                # "actions": []
            }],
            "characters": [
                {"local_character_id": f"C{i+1}", "name_or_description": desc}
                for i, desc in enumerate(characters[:3])
            ],
            "speaker_turns": [
                {"speaker_id": "Unknown", "utterance": d}
                for d in dialogues[:5]
            ],
            "scene_type": "",
            "_fallback": True,
            "_fallback_level": fallback_level
        }
    
    def _extract_summary_from_response(self, response: str) -> str:
        """
        Extract summary-like content from raw response using regex.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            Extracted summary string, empty if not found.
        """
        import re
        
        summary_patterns = [
            r'"clip_summary"\s*:\s*"([^"]+)"',
            r'"summary"\s*:\s*"([^"]+)"',
            r'"sub_clip_summary"\s*:\s*"([^"]+)"',
            r'Summary:\s*([^\n]+)',
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        lines = [l.strip() for l in response.split('\n') if l.strip() and len(l.strip()) > 30]
        if lines:
            cleaned = re.sub(r'[{}\[\]":]', ' ', lines[0])
            return ' '.join(cleaned.split())[:300]
        
        return ""
    
    def _extract_dialogues_from_response(self, response: str) -> List[str]:
        """
        Extract dialogue snippets from raw response using regex.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            List of extracted dialogue strings.
        """
        import re
        
        dialogues = []
        dialogue_patterns = [
            r'"utterance"\s*:\s*"([^"]+)"',
            r'"text"\s*:\s*"([^"]+)"',
            r'"dialogue"\s*:\s*\[\s*"([^"]+)"',
        ]
        
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, response)
            dialogues.extend(matches)
        
        return list(set(dialogues))[:5]
    
    def _extract_characters_from_response(self, response: str) -> List[str]:
        """
        Extract character descriptions from raw response using regex.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            List of extracted character description strings.
        """
        import re
        
        characters = []
        char_patterns = [
            r'"name_or_description"\s*:\s*"([^"]+)"',
            r'"description"\s*:\s*"([^"]+)"',
            r'"actors"\s*:\s*\[\s*"([^"]+)"',
        ]
        
        for pattern in char_patterns:
            matches = re.findall(pattern, response)
            characters.extend(matches)
        
        return list(set(characters))[:5]


class VideoProcessor:
    """
    Main video processor that handles end-to-end clip processing
    and event node creation.
    """
    
    def __init__(
        self,
        video_path: str,
        node_factory: Optional[EventNodeFactory] = None,
        clip_analyzer: Optional[ClipAnalyzer] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the video file.
            node_factory: Optional EventNodeFactory instance.
            clip_analyzer: Optional ClipAnalyzer instance.
        """
        self.video_path = video_path
        self.video_id = Path(video_path).stem
        
        self.node_factory = node_factory or get_node_factory()
        self.clip_analyzer = clip_analyzer or ClipAnalyzer()
        
        self._config = get_pipeline_config()
        
        # Track processed clips
        self.processed_clips: List[str] = []
        self.clip_results: Dict[str, Dict[str, Any]] = {}
    
    def process_clip(
        self,
        clip_id: str,
        start_time: float,
        end_time: float,
        prev_summary: Optional[str] = None
    ) -> List[AdaptiveEventNode]:
        """
        Process a single clip and create event nodes.
        
        Args:
            clip_id: Identifier for this clip.
            start_time: Clip start time in seconds.
            end_time: Clip end time in seconds.
            prev_summary: Summary from previous clip for fallback interpolation.
            
        Returns:
            List of created AdaptiveEventNode instances.
        """
        logger.info(f"Processing clip {clip_id} [{start_time:.1f}s - {end_time:.1f}s]")
        
        # Analyze clip with LLM
        analysis = self.clip_analyzer.analyze_clip(
            video_path=self.video_path,
            start_time=start_time,
            end_time=end_time,
            clip_id=clip_id,
            prev_summary=prev_summary
        )
        
        # Store analysis result
        self.clip_results[clip_id] = analysis
        self.processed_clips.append(clip_id)
        
        # Build character mapping
        characters_map = {}
        for char in analysis.get("characters", []):
            char_id = char.get("local_character_id", "")
            char_desc = char.get("name_or_description", char_id)
            if char_id:
                characters_map[char_id] = char_desc
        
        # Create event nodes
        nodes = []
        scene_type = analysis.get("scene_type", "")
        
        for event_data in analysis.get("events", []):
            node = self.node_factory.create_node_from_llm_output(
                video_id=self.video_id,
                clip_id=clip_id,
                clip_time_start=start_time,
                event_data=event_data,
                scene_type=scene_type,
                characters_map=characters_map
            )
            nodes.append(node)
        
        # If no events extracted, create a single node from clip summary
        if not nodes and analysis.get("clip_summary"):
            node = AdaptiveEventNode(
                node_id=self.node_factory.allocate_id(),
                video_id=self.video_id,
                clip_ids=[clip_id],
                time_start=start_time,
                time_end=end_time,
                summary_text=analysis.get("clip_summary", ""),
                scene_type=scene_type,
                dialogue_snippets=[
                    turn.get("utterance", "")
                    for turn in analysis.get("speaker_turns", [])
                ],
                persons=[
                    char.get("name_or_description", "")
                    for char in analysis.get("characters", [])
                ]
            )
            nodes.append(node)
        
        logger.info(f"Created {len(nodes)} event nodes from clip {clip_id}")
        
        return nodes
    
    def process_all_clips(
        self,
        target_clip_sec: Optional[float] = None,
        max_clips: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> Generator[List[AdaptiveEventNode], None, None]:
        """
        Process all clips from the video sequentially (online streaming).
        
        Maintains a running context of the previous clip's summary for
        fallback interpolation when JSON parsing fails.
        
        Args:
            target_clip_sec: Target clip length in seconds.
            max_clips: Optional maximum number of clips to process.
            callback: Optional callback function called with each batch of nodes.
            
        Yields:
            List of AdaptiveEventNode instances for each clip.
        """
        clip_count = 0
        prev_summary = None
        
        for clip_id, start_time, end_time in generate_clip_segments(
            self.video_path, target_clip_sec
        ):
            if max_clips and clip_count >= max_clips:
                break
            
            try:
                nodes = self.process_clip(
                    clip_id, start_time, end_time,
                    prev_summary=prev_summary
                )
                
                # Update prev_summary for next clip's fallback
                analysis = self.clip_results.get(clip_id, {})
                prev_summary = analysis.get("clip_summary", "")
                
                if callback:
                    callback(nodes)
                
                yield nodes
                
            except Exception as e:
                logger.error(f"Failed to process clip {clip_id}: {e}")
                continue
            
            clip_count += 1
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fallback usage during processing.
        
        Returns:
            Dictionary containing fallback counts and rates.
        """
        total_clips = len(self.clip_results)
        fallback_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for clip_id, result in self.clip_results.items():
            if result.get("_fallback"):
                level = result.get("_fallback_level", 5)
                fallback_counts[level] = fallback_counts.get(level, 0) + 1
            else:
                fallback_counts[0] += 1
        
        return {
            "total_clips": total_clips,
            "normal_parse": fallback_counts[0],
            "json_repair": fallback_counts[1],
            "semantic_description": fallback_counts[2],
            "neighbor_interpolation": fallback_counts[3],
            "regex_extraction": fallback_counts[4],
            "generic_placeholder": fallback_counts[5],
            "fallback_rate": (total_clips - fallback_counts[0]) / total_clips if total_clips > 0 else 0
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clip_analyzer.cleanup()


def process_video_to_nodes(
    video_path: str,
    target_clip_sec: Optional[float] = None,
    max_clips: Optional[int] = None,
    node_factory: Optional[EventNodeFactory] = None
) -> Tuple[List[AdaptiveEventNode], Dict[str, Any]]:
    """
    Convenience function to process a video and return all event nodes.
    
    Args:
        video_path: Path to the video file.
        target_clip_sec: Target clip length in seconds.
        max_clips: Optional maximum number of clips to process.
        node_factory: Optional EventNodeFactory instance.
        
    Returns:
        Tuple of (list of all nodes, processing metadata).
    """
    processor = VideoProcessor(video_path, node_factory=node_factory)
    
    all_nodes = []
    start_time = time.time()
    
    try:
        for nodes in processor.process_all_clips(target_clip_sec, max_clips):
            all_nodes.extend(nodes)
    finally:
        processor.cleanup()
    
    elapsed_time = time.time() - start_time
    
    metadata = {
        "video_path": video_path,
        "video_id": processor.video_id,
        "num_clips_processed": len(processor.processed_clips),
        "num_nodes_created": len(all_nodes),
        "processing_time_sec": elapsed_time,
        "clip_results": processor.clip_results
    }
    
    return all_nodes, metadata


if __name__ == "__main__":
    # Test video processor
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_path> [max_clips]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    max_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Processing video: {video_path}")
    print(f"Max clips: {max_clips}")
    
    nodes, metadata = process_video_to_nodes(video_path, max_clips=max_clips)
    
    print(f"\nResults:")
    print(f"  Clips processed: {metadata['num_clips_processed']}")
    print(f"  Nodes created: {metadata['num_nodes_created']}")
    print(f"  Processing time: {metadata['processing_time_sec']:.2f}s")
    
    for node in nodes:
        print(f"\nNode {node.node_id}:")
        print(f"  Time: {node.time_start:.1f}s - {node.time_end:.1f}s")
        print(f"  Summary: {node.summary_text}")
        print(f"  Persons: {node.persons}")
        print(f"  Objects: {node.objects}")
