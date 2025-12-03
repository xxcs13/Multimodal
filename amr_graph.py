"""
AMR-Graph (Adaptive Multi-Channel Relational Memory Graph) structure.

This module implements the multi-channel graph for storing and navigating
event nodes with temporal, entity jump, and object lifecycle edges.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path

from event_node import AdaptiveEventNode
from config import get_pipeline_config


logger = logging.getLogger(__name__)


@dataclass
class EdgePayload:
    """
    Payload for graph edges containing relationship metadata.
    """
    edge_type: str
    time_gap: float = 0.0
    shared_entities: List[str] = field(default_factory=list)
    action_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_type": self.edge_type,
            "time_gap": self.time_gap,
            "shared_entities": self.shared_entities,
            "action_type": self.action_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgePayload":
        return cls(
            edge_type=data.get("edge_type", ""),
            time_gap=data.get("time_gap", 0.0),
            shared_entities=data.get("shared_entities", []),
            action_type=data.get("action_type", "")
        )


class AMRGraph:
    """
    Adaptive Multi-Channel Relational Memory Graph.
    
    Stores event nodes and maintains multiple edge channels:
    - temporal: Sequential time relationships
    - entity_jump: Non-local entity reappearances
    - object_lifecycle: Object state transitions
    """
    
    def __init__(self, video_id: str = ""):
        """
        Initialize an empty AMR-Graph.
        
        Args:
            video_id: Optional default video ID for this graph.
        """
        self.video_id = video_id
        
        # Node storage: node_id -> AdaptiveEventNode
        self.nodes: Dict[int, AdaptiveEventNode] = {}
        
        # Edge storage by channel
        # Format: {src_id: [(dst_id, EdgePayload), ...]}
        self.edges: Dict[str, Dict[int, List[Tuple[int, EdgePayload]]]] = {
            "temporal": {},
            "entity_jump": {},
            "object_lifecycle": {},
        }
        
        # Index by video_id -> list of node_ids
        self.video_index: Dict[str, List[int]] = {}
        
        # Person index: person_descriptor -> list of node_ids
        self.person_index: Dict[str, List[int]] = {}
        
        # Object index: object_name -> list of node_ids
        self.object_index: Dict[str, List[int]] = {}
        
        # Track the last node for each video (for temporal edges)
        self._last_node_by_video: Dict[str, int] = {}
        
        # Configuration
        self._config = get_pipeline_config()
    
    @property
    def num_nodes(self) -> int:
        """Get the total number of nodes in the graph."""
        return len(self.nodes)
    
    @property
    def num_edges(self) -> Dict[str, int]:
        """Get the count of edges by channel."""
        return {
            channel: sum(len(targets) for targets in edges.values())
            for channel, edges in self.edges.items()
        }
    
    def add_node(self, node: AdaptiveEventNode) -> None:
        """
        Add a node to the graph and update indices.
        
        Args:
            node: The event node to add.
        """
        # Store node
        self.nodes[node.node_id] = node
        
        # Update video index
        if node.video_id not in self.video_index:
            self.video_index[node.video_id] = []
        self.video_index[node.video_id].append(node.node_id)
        
        # Update person index
        for person in node.persons:
            person_key = self._normalize_person_key(person)
            if person_key not in self.person_index:
                self.person_index[person_key] = []
            self.person_index[person_key].append(node.node_id)
        
        # Update object index
        for obj in node.objects:
            obj_key = self._normalize_object_key(obj)
            if obj_key not in self.object_index:
                self.object_index[obj_key] = []
            self.object_index[obj_key].append(node.node_id)
    
    def add_edge(
        self,
        channel: str,
        src_id: int,
        dst_id: int,
        payload: Optional[EdgePayload] = None
    ) -> None:
        """
        Add an edge between two nodes.
        
        Args:
            channel: Edge channel (temporal, entity_jump, object_lifecycle).
            src_id: Source node ID.
            dst_id: Destination node ID.
            payload: Optional edge payload with metadata.
        """
        if channel not in self.edges:
            logger.warning(f"Unknown edge channel: {channel}")
            return
        
        if src_id not in self.nodes or dst_id not in self.nodes:
            logger.warning(f"Cannot add edge: node {src_id} or {dst_id} not found")
            return
        
        if payload is None:
            payload = EdgePayload(edge_type=channel)
        
        if src_id not in self.edges[channel]:
            self.edges[channel][src_id] = []
        
        self.edges[channel][src_id].append((dst_id, payload))
    
    def get_outgoing_edges(
        self,
        node_id: int,
        channels: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[int, EdgePayload]]]:
        """
        Get all outgoing edges from a node.
        
        Args:
            node_id: Source node ID.
            channels: Optional list of channels to query. None means all.
            
        Returns:
            Dictionary mapping channel to list of (dst_id, payload) tuples.
        """
        if channels is None:
            channels = list(self.edges.keys())
        
        result = {}
        for channel in channels:
            if channel in self.edges and node_id in self.edges[channel]:
                result[channel] = self.edges[channel][node_id]
            else:
                result[channel] = []
        
        return result
    
    def get_incoming_edges(
        self,
        node_id: int,
        channels: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[int, EdgePayload]]]:
        """
        Get all incoming edges to a node.
        
        Args:
            node_id: Target node ID.
            channels: Optional list of channels to query. None means all.
            
        Returns:
            Dictionary mapping channel to list of (src_id, payload) tuples.
        """
        if channels is None:
            channels = list(self.edges.keys())
        
        result = {ch: [] for ch in channels}
        
        for channel in channels:
            if channel not in self.edges:
                continue
            for src_id, edges in self.edges[channel].items():
                for dst_id, payload in edges:
                    if dst_id == node_id:
                        result[channel].append((src_id, payload))
        
        return result
    
    def process_new_node(self, node: AdaptiveEventNode) -> None:
        """
        Process a new node: add it to the graph and create appropriate edges.
        
        This is the main entry point for online graph construction.
        
        Args:
            node: The new event node to add.
        """
        # Add node to graph
        self.add_node(node)
        
        # Create temporal edge
        self._create_temporal_edge(node)
        
        # Create entity jump edges
        self._create_entity_jump_edges(node)
        
        # Create object lifecycle edges
        self._create_object_lifecycle_edges(node)
        
        # Update last node tracker
        self._last_node_by_video[node.video_id] = node.node_id
    
    def _create_temporal_edge(self, node: AdaptiveEventNode) -> None:
        """
        Create temporal edge from previous node if exists.
        
        Args:
            node: The new event node.
        """
        video_id = node.video_id
        
        if video_id in self._last_node_by_video:
            prev_node_id = self._last_node_by_video[video_id]
            prev_node = self.nodes[prev_node_id]
            
            time_gap = node.time_start - prev_node.time_end
            
            payload = EdgePayload(
                edge_type="temporal",
                time_gap=time_gap
            )
            
            self.add_edge("temporal", prev_node_id, node.node_id, payload)
    
    def _create_entity_jump_edges(self, node: AdaptiveEventNode) -> None:
        """
        Create entity jump edges for persons reappearing after time gap.
        
        Args:
            node: The new event node.
        """
        threshold = self._config.entity_jump_time_threshold
        
        for person in node.persons:
            person_key = self._normalize_person_key(person)
            
            if person_key not in self.person_index:
                continue
            
            # Find the most recent previous node with this person
            prev_nodes = [
                nid for nid in self.person_index[person_key]
                if nid != node.node_id and nid in self.nodes
            ]
            
            if not prev_nodes:
                continue
            
            # Get the most recent one
            prev_node_id = max(prev_nodes, key=lambda nid: self.nodes[nid].time_end)
            prev_node = self.nodes[prev_node_id]
            
            time_gap = node.time_start - prev_node.time_end
            
            # Only create entity jump if time gap exceeds threshold
            if time_gap > threshold:
                payload = EdgePayload(
                    edge_type="entity_jump",
                    time_gap=time_gap,
                    shared_entities=[person]
                )
                
                self.add_edge("entity_jump", prev_node_id, node.node_id, payload)
    
    def _create_object_lifecycle_edges(self, node: AdaptiveEventNode) -> None:
        """
        Create object lifecycle edges for state-changing actions.
        
        Args:
            node: The new event node.
        """
        state_change_verbs = self._config.state_change_verbs
        
        for action in node.actions:
            verb = action.get("verb", "").lower()
            obj = action.get("object", "")
            
            if not obj:
                continue
            
            # Check if verb indicates state change
            is_state_change = any(sv in verb for sv in state_change_verbs)
            
            if not is_state_change:
                continue
            
            obj_key = self._normalize_object_key(obj)
            
            if obj_key not in self.object_index:
                continue
            
            # Find previous node with this object
            prev_nodes = [
                nid for nid in self.object_index[obj_key]
                if nid != node.node_id and nid in self.nodes
            ]
            
            if not prev_nodes:
                continue
            
            prev_node_id = max(prev_nodes, key=lambda nid: self.nodes[nid].time_end)
            prev_node = self.nodes[prev_node_id]
            
            time_gap = node.time_start - prev_node.time_end
            
            payload = EdgePayload(
                edge_type="object_lifecycle",
                time_gap=time_gap,
                shared_entities=[obj],
                action_type=verb
            )
            
            self.add_edge("object_lifecycle", prev_node_id, node.node_id, payload)
    
    def _normalize_person_key(self, person: str) -> str:
        """Normalize person descriptor for indexing."""
        return person.lower().strip()
    
    def _normalize_object_key(self, obj: str) -> str:
        """Normalize object name for indexing."""
        return obj.lower().strip()
    
    def get_nodes_by_person(self, person: str) -> List[int]:
        """
        Get all node IDs involving a specific person.
        
        Args:
            person: Person descriptor to search for.
            
        Returns:
            List of node IDs.
        """
        person_key = self._normalize_person_key(person)
        return self.person_index.get(person_key, [])
    
    def get_nodes_by_object(self, obj: str) -> List[int]:
        """
        Get all node IDs involving a specific object.
        
        Args:
            obj: Object name to search for.
            
        Returns:
            List of node IDs.
        """
        obj_key = self._normalize_object_key(obj)
        return self.object_index.get(obj_key, [])
    
    def get_nodes_in_time_range(
        self,
        video_id: str,
        start_time: float,
        end_time: float
    ) -> List[int]:
        """
        Get all node IDs within a time range for a video.
        
        Args:
            video_id: Video ID to search in.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            
        Returns:
            List of node IDs ordered by time.
        """
        if video_id not in self.video_index:
            return []
        
        result = []
        for node_id in self.video_index[video_id]:
            node = self.nodes[node_id]
            if node.time_start >= start_time and node.time_end <= end_time:
                result.append(node_id)
        
        return sorted(result, key=lambda nid: self.nodes[nid].time_start)
    
    def get_all_nodes_sorted(self, video_id: Optional[str] = None) -> List[int]:
        """
        Get all node IDs sorted by time.
        
        Args:
            video_id: Optional video ID to filter by.
            
        Returns:
            List of node IDs sorted by time_start.
        """
        if video_id:
            nodes = self.video_index.get(video_id, [])
        else:
            nodes = list(self.nodes.keys())
        
        return sorted(nodes, key=lambda nid: self.nodes[nid].time_start)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the graph.
        """
        # Serialize nodes
        nodes_data = {
            str(nid): node.to_dict() for nid, node in self.nodes.items()
        }
        
        # Serialize edges
        edges_data = {}
        for channel, channel_edges in self.edges.items():
            edges_data[channel] = {}
            for src_id, targets in channel_edges.items():
                edges_data[channel][str(src_id)] = [
                    {"dst_id": dst_id, "payload": payload.to_dict()}
                    for dst_id, payload in targets
                ]
        
        return {
            "video_id": self.video_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "video_index": {k: v for k, v in self.video_index.items()},
            "person_index": {k: v for k, v in self.person_index.items()},
            "object_index": {k: v for k, v in self.object_index.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AMRGraph":
        """
        Create graph from dictionary.
        
        Args:
            data: Dictionary containing graph data.
            
        Returns:
            AMRGraph instance.
        """
        graph = cls(video_id=data.get("video_id", ""))
        
        # Load nodes
        for nid_str, node_data in data.get("nodes", {}).items():
            node = AdaptiveEventNode.from_dict(node_data)
            graph.nodes[node.node_id] = node
        
        # Load edges
        for channel, channel_edges in data.get("edges", {}).items():
            if channel not in graph.edges:
                graph.edges[channel] = {}
            for src_id_str, targets in channel_edges.items():
                src_id = int(src_id_str)
                graph.edges[channel][src_id] = [
                    (t["dst_id"], EdgePayload.from_dict(t["payload"]))
                    for t in targets
                ]
        
        # Load indices
        graph.video_index = data.get("video_index", {})
        graph.person_index = data.get("person_index", {})
        graph.object_index = data.get("object_index", {})
        
        # Rebuild last node tracker
        for vid, node_ids in graph.video_index.items():
            if node_ids:
                graph._last_node_by_video[vid] = max(
                    node_ids, key=lambda nid: graph.nodes[nid].time_end
                )
        
        return graph
    
    def save_to_json(self, filepath: str) -> None:
        """
        Save graph to JSON file.
        
        Args:
            filepath: Path to save the JSON file.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved graph to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> "AMRGraph":
        """
        Load graph from JSON file.
        
        Args:
            filepath: Path to the JSON file.
            
        Returns:
            AMRGraph instance.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with various graph statistics.
        """
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_videos": len(self.video_index),
            "num_unique_persons": len(self.person_index),
            "num_unique_objects": len(self.object_index),
            "nodes_per_video": {
                vid: len(nodes) for vid, nodes in self.video_index.items()
            }
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"AMRGraph(nodes={stats['num_nodes']}, "
            f"edges={stats['num_edges']}, "
            f"videos={stats['num_videos']})"
        )


if __name__ == "__main__":
    # Test AMR-Graph
    from event_node import EventNodeFactory
    
    factory = EventNodeFactory()
    graph = AMRGraph(video_id="test_video")
    
    # Create test nodes
    events = [
        {
            "local_event_id": "E1",
            "time_start": 0.0,
            "time_end": 5.0,
            "summary": "Person A enters the room",
            "actors": ["C1"],
            "objects": [{"name": "door", "spatial_description": "behind person"}],
            "dialogue": [],
            "actions": [{"actor": "C1", "verb": "enter", "object": "room"}]
        },
        {
            "local_event_id": "E2",
            "time_start": 5.0,
            "time_end": 10.0,
            "summary": "Person A picks up a book",
            "actors": ["C1"],
            "objects": [{"name": "book", "spatial_description": "on the table"}],
            "dialogue": ["This is interesting"],
            "actions": [{"actor": "C1", "verb": "pick up", "object": "book"}]
        },
        {
            "local_event_id": "E3",
            "time_start": 60.0,
            "time_end": 65.0,
            "summary": "Person A returns and puts down the book",
            "actors": ["C1"],
            "objects": [{"name": "book", "spatial_description": "on the shelf"}],
            "dialogue": [],
            "actions": [{"actor": "C1", "verb": "put down", "object": "book"}]
        }
    ]
    
    characters_map = {"C1": "person in blue shirt"}
    
    for i, event in enumerate(events):
        node = factory.create_node_from_llm_output(
            video_id="test_video",
            clip_id=f"clip_{i}",
            clip_time_start=event["time_start"],
            event_data=event,
            characters_map=characters_map
        )
        graph.process_new_node(node)
    
    print("Graph statistics:")
    print(json.dumps(graph.get_statistics(), indent=2))
    
    print("\nEdges:")
    for channel, channel_edges in graph.edges.items():
        print(f"  {channel}:")
        for src, targets in channel_edges.items():
            for dst, payload in targets:
                print(f"    {src} -> {dst}: {payload.to_dict()}")
    
    # Test serialization
    graph.save_to_json("/tmp/test_graph.json")
    loaded_graph = AMRGraph.load_from_json("/tmp/test_graph.json")
    print(f"\nLoaded graph: {loaded_graph}")
