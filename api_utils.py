"""
API utilities for OpenRouter LLM and embedding calls.

This module provides functions for making API calls to OpenRouter,
including text generation, multimodal analysis, and embedding generation.
"""

import os
import time
import json
import base64
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

import openai
from openai import OpenAI

from config import get_api_config, get_pipeline_config


# Configure logging
logger = logging.getLogger(__name__)


def get_client(model_name: str) -> OpenAI:
    """
    Get OpenAI client configured for the specified model.
    
    Args:
        model_name: Name of the model (e.g., "openai/gpt-4o").
        
    Returns:
        Configured OpenAI client.
    """
    api_config = get_api_config()
    
    if model_name not in api_config:
        raise ValueError(f"Model '{model_name}' not found in API config")
    
    config = api_config[model_name]
    
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"]
    )


def call_llm(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None
) -> Tuple[str, int]:
    """
    Make a single LLM call with the specified model.
    
    Args:
        model_name: Name of the model to use.
        messages: List of message dictionaries for the conversation.
        temperature: Optional temperature override.
        max_tokens: Optional max tokens override.
        response_format: Optional response format (e.g., {"type": "json_object"}).
        
    Returns:
        Tuple of (response text, total tokens used).
    """
    config = get_pipeline_config()
    client = get_client(model_name)
    
    if temperature is None:
        temperature = config.llm_temperature
    if max_tokens is None:
        max_tokens = config.llm_max_tokens
    
    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if response_format:
        kwargs["response_format"] = response_format
    
    response = client.chat.completions.create(**kwargs)
    
    content = response.choices[0].message.content or ""
    total_tokens = response.usage.total_tokens if response.usage else 0
    
    return content, total_tokens


def call_llm_with_retry(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = None,
    retry_count: Optional[int] = None,
    retry_delay: Optional[float] = None
) -> Tuple[str, int]:
    """
    Make an LLM call with automatic retry on failure.
    
    Args:
        model_name: Name of the model to use.
        messages: List of message dictionaries.
        temperature: Optional temperature override.
        max_tokens: Optional max tokens override.
        response_format: Optional response format.
        retry_count: Optional number of retries.
        retry_delay: Optional delay between retries in seconds.
        
    Returns:
        Tuple of (response text, total tokens used).
        
    Raises:
        Exception: If all retries fail.
    """
    config = get_pipeline_config()
    
    if retry_count is None:
        retry_count = config.llm_retry_count
    if retry_delay is None:
        retry_delay = config.llm_retry_delay
    
    last_error = None
    
    for attempt in range(retry_count + 1):
        try:
            return call_llm(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
        except Exception as e:
            last_error = e
            logger.warning(
                f"LLM call attempt {attempt + 1}/{retry_count + 1} failed: {e}"
            )
            if attempt < retry_count:
                time.sleep(retry_delay)
    
    raise Exception(
        f"Failed to get LLM response after {retry_count + 1} attempts. "
        f"Last error: {last_error}"
    )


def build_text_message(text: str, role: str = "user") -> Dict[str, Any]:
    """
    Build a simple text message for LLM conversation.
    
    Args:
        text: The text content.
        role: Message role (user, assistant, system).
        
    Returns:
        Message dictionary.
    """
    return {
        "role": role,
        "content": text
    }


def build_multimodal_message(
    text: str,
    images: Optional[List[str]] = None,
    video_base64: Optional[str] = None,
    role: str = "user"
) -> Dict[str, Any]:
    """
    Build a multimodal message with text, images, and/or video.
    
    Args:
        text: The text prompt.
        images: Optional list of image URLs or base64 strings.
        video_base64: Optional base64-encoded video.
        role: Message role.
        
    Returns:
        Message dictionary with multimodal content.
    """
    content = []
    
    # Add images if provided
    if images:
        for img in images:
            if img.startswith("http"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            else:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
    
    # Add video if provided
    if video_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:video/mp4;base64,{video_base64}"}
        })
    
    # Add text
    content.append({
        "type": "text",
        "text": text
    })
    
    return {
        "role": role,
        "content": content
    }


def get_embedding(
    model_name: str,
    text: Union[str, List[str]]
) -> List[List[float]]:
    """
    Get embeddings for text using the specified model.
    
    Args:
        model_name: Name of the embedding model.
        text: Single text string or list of texts.
        
    Returns:
        List of embedding vectors.
        
    Raises:
        Exception: If embedding call fails or returns no data.
    """
    client = get_client(model_name)
    
    if isinstance(text, str):
        text = [text]
    
    # Filter out empty or whitespace-only texts
    cleaned_texts = []
    empty_indices = []
    for i, t in enumerate(text):
        if t and t.strip():
            cleaned_texts.append(t.strip())
        else:
            empty_indices.append(i)
            logger.warning(f"Empty text at index {i}, will use placeholder")
    
    # If all texts are empty, raise error
    if not cleaned_texts:
        raise Exception("All input texts are empty - cannot generate embeddings")
    
    response = client.embeddings.create(
        model=model_name,
        input=cleaned_texts
    )
    
    # Check if we got embeddings back
    if not response.data:
        raise Exception(f"No embedding data received from API. Input had {len(cleaned_texts)} texts.")
    
    if len(response.data) != len(cleaned_texts):
        raise Exception(
            f"Mismatch in embedding count: got {len(response.data)}, expected {len(cleaned_texts)}"
        )
    
    embeddings = [item.embedding for item in response.data]
    
    # Re-insert placeholder embeddings for empty texts (zeros)
    if empty_indices:
        dim = len(embeddings[0]) if embeddings else 3072
        for idx in empty_indices:
            embeddings.insert(idx, [0.0] * dim)
            logger.warning(f"Inserted zero embedding for empty text at index {idx}")
    
    return embeddings


def get_embedding_with_retry(
    model_name: str,
    text: Union[str, List[str]],
    retry_count: Optional[int] = None,
    retry_delay: Optional[float] = None
) -> List[List[float]]:
    """
    Get embeddings with automatic retry on failure.
    
    Args:
        model_name: Name of the embedding model.
        text: Single text string or list of texts.
        retry_count: Optional number of retries.
        retry_delay: Optional delay between retries.
        
    Returns:
        List of embedding vectors.
    """
    config = get_pipeline_config()
    
    if retry_count is None:
        retry_count = config.llm_retry_count
    if retry_delay is None:
        retry_delay = config.llm_retry_delay
    
    last_error = None
    
    for attempt in range(retry_count + 1):
        try:
            return get_embedding(model_name, text)
        except Exception as e:
            last_error = e
            logger.warning(
                f"Embedding call attempt {attempt + 1}/{retry_count + 1} failed: {e}"
            )
            if attempt < retry_count:
                time.sleep(retry_delay)
    
    raise Exception(
        f"Failed to get embeddings after {retry_count + 1} attempts. "
        f"Last error: {last_error}"
    )


def video_to_base64(video_path: str) -> str:
    """
    Convert a video file to base64 string.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Base64-encoded string of the video.
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode("utf-8")


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 string.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Base64-encoded string of the image.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling common formatting issues.
    
    This function attempts multiple cleanup strategies to handle malformed JSON
    from LLM responses, including trailing commas, unescaped characters, and
    incomplete structures.
    
    Args:
        response: Raw response string from LLM.
        
    Returns:
        Parsed JSON dictionary.
        
    Raises:
        json.JSONDecodeError: If parsing fails after all cleanup attempts.
    """
    import re
    
    # Remove markdown code fences if present
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try to find JSON object boundaries
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx + 1]
    
    # Attempt 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Fix trailing commas (common LLM error)
    # Remove trailing commas before ] or }
    cleaned = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Attempt 3: Fix unescaped newlines in strings
    # Replace actual newlines inside strings with \n
    def fix_newlines(match):
        s = match.group(0)
        # Replace actual newlines with escaped newlines
        s = s.replace('\n', '\\n').replace('\r', '\\r')
        return s
    
    # Match strings and fix newlines within them
    cleaned2 = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_newlines, cleaned)
    try:
        return json.loads(cleaned2)
    except json.JSONDecodeError:
        pass
    
    # Attempt 4: Try to fix truncated JSON by finding balanced braces
    def find_balanced_json(s):
        """Find the largest balanced JSON object in a string."""
        depth = 0
        start = s.find('{')
        if start == -1:
            return None
        
        for i, c in enumerate(s[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        
        # If unbalanced, try to close it
        if depth > 0:
            return s[start:] + '}' * depth
        return None
    
    balanced = find_balanced_json(cleaned2)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass
    
    # Attempt 5: Remove control characters
    cleaned3 = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned2)
    try:
        return json.loads(cleaned3)
    except json.JSONDecodeError:
        pass
    
    # Final attempt: raise with original text for debugging
    raise json.JSONDecodeError(
        f"Failed to parse JSON after all cleanup attempts",
        text[:500],  # Truncate for readability
        0
    )


def parse_json_list_response(response: str) -> List[Any]:
    """
    Parse JSON list from LLM response, handling common formatting issues.
    
    Args:
        response: Raw response string from LLM.
        
    Returns:
        Parsed JSON list.
        
    Raises:
        json.JSONDecodeError: If parsing fails after cleanup.
    """
    # Remove markdown code fences if present
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try to find JSON array boundaries
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx + 1]
    
    return json.loads(text)

