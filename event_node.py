"""
Event node structure for AMR-Graph.

This module defines the AdaptiveEventNode class which represents
individual events extracted from video clips, along with their
metadata, embeddings, and relationships.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AdaptiveEventNode:
    """
    Represents a single event extracted from a video clip.
    
    This node captures all relevant information about an event including
    temporal boundaries, involved actors, objects, actions, dialogue,
    and the computed text embedding for retrieval.
    """
    
    # Core identifiers
    node_id: int
    video_id: str
    clip_ids: List[str] = field(default_factory=list)
    
    # Temporal information
    time_start: float = 0.0
    time_end: float = 0.0
    
    # Content
    summary_text: str = ""
    dialogue_snippets: List[str] = field(default_factory=list)
    
    # Entity information
    persons: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    scene_type: str = ""
    
    # Action information
    actions: List[Dict[str, str]] = field(default_factory=list)
    
    # Spatial descriptions for objects
    object_spatial_info: List[Dict[str, str]] = field(default_factory=list)
    
    # Embedding (stored as list for JSON serialization)
    text_embedding: Optional[List[float]] = None
    
    # Raw LLM output for debugging
    raw_llm_output: Optional[Dict[str, Any]] = None
    
    def get_embedding_array(self) -> Optional[np.ndarray]:
        """
        Get the text embedding as a numpy array.
        
        Returns:
            Numpy array of the embedding or None if not set.
        """
        if self.text_embedding is None:
            return None
        return np.array(self.text_embedding, dtype=np.float32)
    
    def set_embedding_from_array(self, embedding: np.ndarray) -> None:
        """
        Set the text embedding from a numpy array.
        
        Args:
            embedding: Numpy array of the embedding.
        """
        self.text_embedding = embedding.tolist()
    
    def get_searchable_text(self) -> str:
        """
        Get concatenated text for search indexing (BM25).
        
        Returns:
            Combined text from summary, dialogue, persons, and objects.
        """
        parts = [self.summary_text]
        
        if self.dialogue_snippets:
            parts.append(" ".join(self.dialogue_snippets))
        
        if self.persons:
            parts.append(" ".join(self.persons))
        
        if self.objects:
            parts.append(" ".join(self.objects))
        
        return " ".join(parts)
    
    def get_action_verbs(self) -> List[str]:
        """
        Extract all action verbs from this event.
        
        Returns:
            List of verb strings.
        """
        verbs = []
        for action in self.actions:
            if isinstance(action, dict):
                verb = action.get("verb", "")
                if verb:
                    verbs.append(verb)
            elif isinstance(action, str) and action:
                verbs.append(action)
        return verbs
    
    def get_action_objects(self) -> List[str]:
        """
        Extract all objects involved in actions.
        
        Returns:
            List of object strings from actions.
        """
        objects = []
        for action in self.actions:
            if isinstance(action, dict):
                obj = action.get("object", "")
                if obj:
                    objects.append(obj)
        return objects
    
    def has_state_change_action(self, state_change_verbs: set) -> bool:
        """
        Check if this event contains any state-changing actions.
        
        Args:
            state_change_verbs: Set of verbs indicating state changes.
            
        Returns:
            True if any action verb indicates a state change.
        """
        for action in self.actions:
            if isinstance(action, dict):
                verb = action.get("verb", "").lower()
            elif isinstance(action, str):
                verb = action.lower()
            else:
                continue
            for sv in state_change_verbs:
                if sv in verb:
                    return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "video_id": self.video_id,
            "clip_ids": self.clip_ids,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "summary_text": self.summary_text,
            "dialogue_snippets": self.dialogue_snippets,
            "persons": self.persons,
            "objects": self.objects,
            "scene_type": self.scene_type,
            "actions": self.actions,
            "object_spatial_info": self.object_spatial_info,
            "text_embedding": self.text_embedding,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveEventNode":
        """
        Create node from dictionary.
        
        Args:
            data: Dictionary containing node data.
            
        Returns:
            AdaptiveEventNode instance.
        """
        return cls(
            node_id=data.get("node_id", 0),
            video_id=data.get("video_id", ""),
            clip_ids=data.get("clip_ids", []),
            time_start=data.get("time_start", 0.0),
            time_end=data.get("time_end", 0.0),
            summary_text=data.get("summary_text", ""),
            dialogue_snippets=data.get("dialogue_snippets", []),
            persons=data.get("persons", []),
            objects=data.get("objects", []),
            scene_type=data.get("scene_type", ""),
            actions=data.get("actions", []),
            object_spatial_info=data.get("object_spatial_info", []),
            text_embedding=data.get("text_embedding"),
        )
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveEventNode(id={self.node_id}, "
            f"video={self.video_id}, "
            f"time=[{self.time_start:.1f}-{self.time_end:.1f}], "
            f"summary='{self.summary_text[:50]}...')"
        )


class EventNodeFactory:
    """
    Factory class for creating and managing event nodes.
    
    Maintains a global counter for node IDs and provides methods
    for constructing nodes from LLM outputs.
    """
    
    def __init__(self, start_id: int = 0):
        """
        Initialize the factory.
        
        Args:
            start_id: Starting value for node ID counter.
        """
        self._next_id = start_id
    
    @property
    def next_id(self) -> int:
        """Get the next available node ID without incrementing."""
        return self._next_id
    
    def allocate_id(self) -> int:
        """
        Allocate and return the next node ID.
        
        Returns:
            The allocated node ID.
        """
        node_id = self._next_id
        self._next_id += 1
        return node_id
    
    def reset(self, start_id: int = 0) -> None:
        """
        Reset the node ID counter.
        
        Args:
            start_id: New starting value for node ID counter.
        """
        self._next_id = start_id
    
    def create_node_from_llm_output(
        self,
        video_id: str,
        clip_id: str,
        clip_time_start: float,
        event_data: Dict[str, Any],
        scene_type: str = "",
        characters_map: Optional[Dict[str, str]] = None
    ) -> AdaptiveEventNode:
        """
        Create an event node from LLM-parsed event data.
        
        Args:
            video_id: ID of the source video.
            clip_id: ID of the clip containing this event.
            clip_time_start: Start time of the clip in the video.
            event_data: Event dictionary from LLM output.
            scene_type: Scene type from clip analysis.
            characters_map: Optional mapping of local character IDs to descriptions.
            
        Returns:
            New AdaptiveEventNode instance.
        """
        if characters_map is None:
            characters_map = {}
        
        # Extract basic timing
        event_time_start = event_data.get("time_start", 0.0)
        event_time_end = event_data.get("time_end", event_time_start)
        
        # Adjust to absolute video time
        abs_time_start = clip_time_start + event_time_start
        abs_time_end = clip_time_start + event_time_end
        
        # Extract actors and resolve character IDs
        actors = event_data.get("actors", [])
        persons = []
        for actor in actors:
            if actor in characters_map:
                persons.append(characters_map[actor])
            else:
                persons.append(actor)
        
        # Extract objects
        objects_data = event_data.get("objects", [])
        objects = []
        object_spatial_info = []
        for obj in objects_data:
            if isinstance(obj, dict):
                objects.append(obj.get("name", ""))
                object_spatial_info.append({
                    "name": obj.get("name", ""),
                    "spatial_description": obj.get("spatial_description", "")
                })
            else:
                objects.append(str(obj))
        
        # Extract dialogue
        dialogue = event_data.get("dialogue", [])
        
        # Extract actions
        actions = event_data.get("actions", [])
        
        # Get event summary
        summary = event_data.get("summary", "")
        
        # Create node
        node = AdaptiveEventNode(
            node_id=self.allocate_id(),
            video_id=video_id,
            clip_ids=[clip_id],
            time_start=abs_time_start,
            time_end=abs_time_end,
            summary_text=summary,
            dialogue_snippets=dialogue,
            persons=persons,
            objects=objects,
            scene_type=scene_type,
            actions=actions,
            object_spatial_info=object_spatial_info,
            raw_llm_output=event_data
        )
        
        return node
    
    def create_merged_node(
        self,
        nodes: List[AdaptiveEventNode],
        new_summary: str,
        new_embedding: Optional[np.ndarray] = None
    ) -> AdaptiveEventNode:
        """
        Create a new node by merging multiple existing nodes.
        
        Args:
            nodes: List of nodes to merge.
            new_summary: New synthesized summary for merged node.
            new_embedding: Optional new embedding for merged node.
            
        Returns:
            New merged AdaptiveEventNode instance.
        """
        if not nodes:
            raise ValueError("Cannot merge empty list of nodes")
        
        # Collect clip IDs
        clip_ids = []
        for node in nodes:
            clip_ids.extend(node.clip_ids)
        clip_ids = list(set(clip_ids))
        
        # Get time range
        time_start = min(node.time_start for node in nodes)
        time_end = max(node.time_end for node in nodes)
        
        # Union of persons and objects
        persons = list(set(p for node in nodes for p in node.persons))
        objects = list(set(o for node in nodes for o in node.objects))
        
        # Collect dialogue
        dialogue = []
        for node in nodes:
            dialogue.extend(node.dialogue_snippets)
        
        # Collect actions
        actions = []
        for node in nodes:
            actions.extend(node.actions)
        
        # Collect spatial info
        object_spatial_info = []
        for node in nodes:
            object_spatial_info.extend(node.object_spatial_info)
        
        # Get scene type (use most common or first)
        scene_types = [node.scene_type for node in nodes if node.scene_type]
        scene_type = scene_types[0] if scene_types else ""
        
        # Create merged node
        merged_node = AdaptiveEventNode(
            node_id=self.allocate_id(),
            video_id=nodes[0].video_id,
            clip_ids=clip_ids,
            time_start=time_start,
            time_end=time_end,
            summary_text=new_summary,
            dialogue_snippets=dialogue,
            persons=persons,
            objects=objects,
            scene_type=scene_type,
            actions=actions,
            object_spatial_info=object_spatial_info,
        )
        
        if new_embedding is not None:
            merged_node.set_embedding_from_array(new_embedding)
        
        return merged_node


# Global factory instance
_node_factory: Optional[EventNodeFactory] = None


def get_node_factory() -> EventNodeFactory:
    """
    Get the global event node factory.
    
    Returns:
        EventNodeFactory instance.
    """
    global _node_factory
    if _node_factory is None:
        _node_factory = EventNodeFactory()
    return _node_factory


def reset_node_factory(start_id: int = 0) -> None:
    """
    Reset the global node factory.
    
    Args:
        start_id: New starting ID for the factory.
    """
    global _node_factory
    _node_factory = EventNodeFactory(start_id)


if __name__ == "__main__":
    # Test event node creation
    factory = EventNodeFactory()
    
    # Test creating node from LLM output
    event_data = {
        "local_event_id": "E1",
        "time_start": 2.0,
        "time_end": 8.0,
        "summary": "A person picks up a book from the table",
        "actors": ["C1"],
        "objects": [
            {"name": "book", "spatial_description": "on the table"},
            {"name": "table", "spatial_description": "in front of the person"}
        ],
        "dialogue": ["I need this book"],
        "actions": [
            {"actor": "C1", "verb": "pick up", "object": "book", "spatial_relation": "from the table"}
        ]
    }
    
    characters_map = {"C1": "person in blue shirt"}
    
    node = factory.create_node_from_llm_output(
        video_id="test_video",
        clip_id="clip_0",
        clip_time_start=10.0,
        event_data=event_data,
        scene_type="indoor_living",
        characters_map=characters_map
    )
    
    print("Created node:")
    print(f"  ID: {node.node_id}")
    print(f"  Time: {node.time_start} - {node.time_end}")
    print(f"  Summary: {node.summary_text}")
    print(f"  Persons: {node.persons}")
    print(f"  Objects: {node.objects}")
    print(f"  Actions: {node.actions}")
    print(f"  Searchable text: {node.get_searchable_text()}")
    
    # Test serialization
    node_dict = node.to_dict()
    restored_node = AdaptiveEventNode.from_dict(node_dict)
    print(f"\nRestored node: {restored_node}")
