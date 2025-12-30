"""
Qwen2.5-Omni-3B provider module for multimodal video analysis.

This module provides functions for loading and using the Qwen2.5-Omni-3B model
for video clip analysis with support for visual, audio, and text modalities.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch

# Configure logging
logger = logging.getLogger(__name__)

# Global model and processor instances for caching
_qwen_model = None
_qwen_processor = None
_model_loaded = False


def get_processing_config() -> Dict[str, Any]:
    """
    Load Qwen processing configuration from JSON file.
    
    Returns:
        Dictionary containing processing configuration.
    """
    config_path = Path(__file__).parent / "configs" / "processing_config.json"
    
    if not config_path.exists():
        # Return default configuration if file not found
        logger.warning(f"Processing config not found at {config_path}, using defaults")
        return {
            "ckpt": "Qwen/Qwen2.5-Omni-3B",
            "temperature": 0.1,
            "max_retries": 3,
            "use_audio_in_video": True,
            "use_flash_attention": True,
            "max_new_tokens": 4096
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_qwen_model() -> Tuple[Any, Any]:
    """
    Get or lazily load the Qwen2.5-Omni model and processor.
    
    Uses global caching to avoid reloading the model on each call.
    
    Returns:
        Tuple of (model, processor).
    """
    global _qwen_model, _qwen_processor, _model_loaded
    
    if _model_loaded:
        return _qwen_model, _qwen_processor
    
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor
    )
    
    config = get_processing_config()
    model_path = config.get("ckpt", "Qwen/Qwen2.5-Omni-3B")
    use_flash_attention = config.get("use_flash_attention", True)
    
    logger.info(f"Loading Qwen2.5-Omni model from {model_path}")
    
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    
    # Check if Flash Attention 2 is available
    if use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2 for acceleration")
        except ImportError:
            logger.warning(
                "Flash Attention 2 requested but flash_attn not installed. "
                "Falling back to default attention implementation."
            )
    
    _qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    _qwen_model.eval()
    
    # Disable audio output to save GPU memory (we only need text)
    _qwen_model.disable_talker()
    logger.info("Disabled talker module to save GPU memory")
    
    _qwen_processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    _model_loaded = True
    logger.info("Qwen2.5-Omni model loaded successfully")
    
    return _qwen_model, _qwen_processor


def build_qwen_messages(
    video_path: str,
    prompt: str,
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Build message list for Qwen chat completion.
    
    Args:
        video_path: Path to the video file.
        prompt: The text prompt to send with the video.
        system_prompt: Optional system prompt override.
        
    Returns:
        List of message dictionaries in Qwen format.
    """
    if system_prompt is None:
        system_prompt = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    return messages


def analyze_video_with_qwen(
    video_path: str,
    prompt: str,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None
) -> Tuple[str, int]:
    """
    Analyze a video clip using Qwen2.5-Omni model.
    
    This function processes the video with audio support and returns
    text analysis. It is compatible with the ClipAnalyzer interface.
    
    Args:
        video_path: Path to the video file to analyze.
        prompt: The analysis prompt to send with the video.
        temperature: Optional temperature for generation.
        max_new_tokens: Optional maximum number of tokens to generate.
        
    Returns:
        Tuple of (response text, token count).
    """
    from qwen_omni_utils import process_mm_info
    
    config = get_processing_config()
    
    if temperature is None:
        temperature = config.get("temperature", 0.1)
    if max_new_tokens is None:
        max_new_tokens = config.get("max_new_tokens", 4096)
    
    use_audio_in_video = config.get("use_audio_in_video", True)
    
    # Get model and processor
    model, processor = get_qwen_model()
    
    # Build messages
    messages = build_qwen_messages(video_path, prompt)
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Process multimedia information
    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=use_audio_in_video
    )
    
    # Prepare inputs
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video
    )
    inputs = inputs.to(model.device).to(model.dtype)
    
    # Generate response (text only, no audio output)
    with torch.no_grad():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
        
        # Decode the generated tokens
        generate_ids = text_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        token_count = len(text_ids[0])
    
    # Clean up GPU memory
    del text_ids
    del generate_ids
    del inputs
    torch.cuda.empty_cache()
    
    return response, token_count


def get_response(messages: List[Dict[str, Any]]) -> Tuple[str, int]:
    """
    Get chat completion response from Qwen model.
    
    This function provides backward compatibility with the original interface.
    
    Args:
        messages: List of message dictionaries in Qwen format.
        
    Returns:
        Tuple of (response content, total tokens used).
    """
    from qwen_omni_utils import process_mm_info
    
    config = get_processing_config()
    temperature = config.get("temperature", 0.1)
    max_new_tokens = config.get("max_new_tokens", 4096)
    use_audio_in_video = config.get("use_audio_in_video", True)
    
    model, processor = get_qwen_model()
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Process multimedia information
    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=use_audio_in_video
    )
    
    # Prepare inputs
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video
    )
    inputs = inputs.to(model.device).to(model.dtype)
    
    # Generate response
    with torch.no_grad():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
        
        generate_ids = text_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        token_count = len(text_ids[0])
    
    # Clean up
    del text_ids
    del generate_ids
    del inputs
    torch.cuda.empty_cache()
    
    return response, token_count


def generate_messages(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate message list for chat completion from mixed inputs.
    
    This function provides backward compatibility with the original interface.
    
    Args:
        inputs: List of input dictionaries with 'type' and 'content' keys.
            Supported types:
            - "text": text content
            - "images/jpeg", "images/png": base64 encoded images
            - "video_url", "video_base64/mp4", "video_base64/webm": video content
            
    Returns:
        List of formatted messages for Qwen chat completion.
    """
    content = []
    
    for item in inputs:
        if not item.get("content"):
            logger.warning("Empty content in input, skipping")
            continue
            
        input_type = item["type"]
        
        if input_type == "text":
            content.append({"type": "text", "text": item["content"]})
            
        elif input_type in ["images/jpeg", "images/png"]:
            # Handle base64 encoded images
            if isinstance(item["content"], list):
                for img in item["content"]:
                    if isinstance(img, str):
                        content.append({
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{img}"
                        })
                    elif isinstance(img, (list, tuple)) and len(img) >= 2:
                        # Handle (caption, base64) format
                        content.append({"type": "text", "text": img[0]})
                        content.append({
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{img[1]}"
                        })
            else:
                content.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{item['content']}"
                })
                
        elif input_type in ["video_url", "video_base64/mp4", "video_base64/webm"]:
            content.append({
                "type": "video",
                "video": item["content"]
            })
            
        else:
            logger.warning(f"Unknown input type: {input_type}, skipping")
    
    if not content:
        raise ValueError("No valid content found in inputs")
    
    messages = [{"role": "user", "content": content}]
    return messages


def cleanup_model() -> None:
    """
    Clean up the loaded model and free GPU memory.
    """
    global _qwen_model, _qwen_processor, _model_loaded
    
    if _qwen_model is not None:
        del _qwen_model
        _qwen_model = None
    
    if _qwen_processor is not None:
        del _qwen_processor
        _qwen_processor = None
    
    _model_loaded = False
    
    torch.cuda.empty_cache()
    logger.info("Qwen model cleaned up and GPU memory freed")