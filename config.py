"""
Configuration management module for AMR-Graph pipeline.

This module handles loading API configurations, environment variables,
and pipeline settings from JSON and .env files.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()


def load_env_variables() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dictionary containing environment variables.
    """
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    
    return {
        "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    }


def load_api_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load API configuration from JSON file.
    
    Args:
        config_path: Optional path to config file. Defaults to configs/api_config.json.
        
    Returns:
        Dictionary containing API configurations for each model.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "api_config.json"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Load environment variables for API key substitution
    env_vars = load_env_variables()
    
    # Replace placeholder API keys with actual values from environment
    for model_name, model_config in config.items():
        if model_config.get("api_key") == "OPENROUTER_API_KEY":
            model_config["api_key"] = env_vars["OPENROUTER_API_KEY"]
        if "base_url" not in model_config:
            model_config["base_url"] = env_vars["OPENROUTER_BASE_URL"]
    
    return config


class PipelineConfig:
    """
    Configuration class for the AMR-Graph pipeline.
    
    Contains all configurable parameters for video processing,
    graph construction, retrieval, and evaluation.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline configuration.
        
        Args:
            config_dict: Optional dictionary to override default settings.
        """
        # Video processing settings
        self.target_clip_sec: float = 30.0  # Changed from 10s to 30s for better context
        self.frame_sample_rate: int = 1  # Sample 1 frame per second for LLM analysis
        
        # Event merging settings
        self.merge_events_enabled: bool = False  # Disable by default for simplicity
        self.merge_time_gap_threshold: float = 5.0
        self.merge_sim_threshold: float = 0.85
        
        # Graph edge settings
        self.entity_jump_time_threshold: float = 30.0  # Time gap for entity jump edges
        self.state_change_verbs: set = {
            "pick up", "put down", "open", "close", "move", "place",
            "take", "grab", "drop", "lift", "set", "arrange", "remove"
        }
        
        # Retrieval settings
        self.anchor_topk_sparse: int = 10  # BM25 top-k
        self.anchor_topk_dense: int = 10   # FAISS top-k
        self.anchor_fusion_k: int = 5      # Final anchors after RRF
        self.navigation_beam_width: int = 20
        self.navigation_max_hops: int = 3
        self.max_visited_nodes: int = 50
        
        # Multimodal model provider selection for clip analysis
        # Options: "qwen" (local Qwen2.5-Omni-3B) or "gemini" (OpenRouter API)
        self.multimodal_provider: str = "qwen"
        
        # Qwen-specific settings (used when multimodal_provider is "qwen")
        self.qwen_model_path: str = "Qwen/Qwen2.5-Omni-3B"
        self.qwen_use_audio_in_video: bool = True
        self.qwen_use_flash_attention: bool = True
        self.qwen_temperature: float = 0.1
        self.qwen_max_new_tokens: int = 4096
        
        # Gemini-specific settings (used when multimodal_provider is "gemini")
        self.gemini_model: str = "google/gemini-2.5-pro"
        
        # API LLM models for other tasks
        self.llm_model_summarization: str = "openai/gpt-4o"
        self.llm_model_question_typing: str = "openai/gpt-4o-mini"
        self.llm_model_verification: str = "openai/gpt-4o"
        self.llm_model_final_answer: str = "openai/gpt-5"
        self.embedding_model: str = "openai/text-embedding-3-large"
        self.embedding_dim: int = 3072  # text-embedding-3-large default; auto-detected if 0
        
        # LLM call settings
        self.llm_temperature: float = 0.1
        self.llm_max_tokens: int = 4096
        self.llm_retry_count: int = 3
        self.llm_retry_delay: float = 5.0
        
        # Search and retrieval rounds
        self.max_retrieval_steps: int = 10
        self.retrieval_topk: int = 5
        
        # Output settings
        self.save_memory_graph: bool = True
        self.memory_graph_dir: str = str(PROJECT_ROOT / "data" / "memory_graphs")
        self.results_dir: str = str(PROJECT_ROOT / "data" / "results")
        self.logs_dir: str = str(PROJECT_ROOT / "data" / "logs")
        
        # Apply custom config if provided
        if config_dict:
            self.update(config_dict)
    
    def get_clip_analysis_model(self) -> str:
        """
        Get the model name for clip analysis based on multimodal_provider setting.
        
        Returns:
            Model name string for clip analysis.
        """
        if self.multimodal_provider == "qwen":
            return self.qwen_model_path
        else:
            return self.gemini_model
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration updates.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary containing all configuration values.
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
    
    @classmethod
    def from_json(cls, json_path: str) -> "PipelineConfig":
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file.
            
        Returns:
            PipelineConfig instance.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def save_to_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration file.
        """
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)


# Global configuration instances
_api_config: Optional[Dict[str, Any]] = None
_pipeline_config: Optional[PipelineConfig] = None


def get_api_config() -> Dict[str, Any]:
    """
    Get the global API configuration.
    
    Returns:
        API configuration dictionary.
    """
    global _api_config
    if _api_config is None:
        _api_config = load_api_config()
    return _api_config


def get_pipeline_config() -> PipelineConfig:
    """
    Get the global pipeline configuration.
    
    Returns:
        PipelineConfig instance.
    """
    global _pipeline_config
    if _pipeline_config is None:
        _pipeline_config = PipelineConfig()
    return _pipeline_config


def set_pipeline_config(config: PipelineConfig) -> None:
    """
    Set the global pipeline configuration.
    
    Args:
        config: PipelineConfig instance to set as global.
    """
    global _pipeline_config
    _pipeline_config = config

