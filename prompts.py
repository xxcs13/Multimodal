"""
Prompt templates for AMR-Graph pipeline.

This module contains all prompt templates used for video analysis,
event summarization, question classification, retrieval, and answer generation.
All prompts are designed to be general and not task-specific.
"""

# =============================================================================
# Video Clip Analysis Prompts
# =============================================================================

PROMPT_CLIP_ANALYSIS = """You are analyzing a video clip to extract structured information about events, characters, dialogue, and objects.

Analyze this video clip and return a JSON object with the following structure. Focus on what is visible and audible in the clip.

Required JSON structure:
{
  "clip_summary": "A brief 1-2 sentence summary of the main activity in this clip",
  "scene_type": "The type of scene (e.g., indoor_living, outdoor_street, office, kitchen, bedroom, etc.)",
  "characters": [
    {
      "local_character_id": "C1",
      "role": "speaker/observer/other",
      "name_or_description": "Description of appearance or name if mentioned"
    }
  ],
  "speaker_turns": [
    {
      "speaker_id": "S1",
      "utterance": "The spoken text",
      "time_start": 0.0,
      "time_end": 5.0
    }
  ],
  "events": [
    {
      "local_event_id": "E1",
      "time_start": 0.0,
      "time_end": 10.0,
      "summary": "What happened in this event",
      "actors": ["C1"],
      "objects": [
        {
          "name": "object name",
          "spatial_description": "relative position using terms like: above/below, left of/right of, in front of/behind, on top of/next to"
        }
      ],
      "dialogue": ["Any relevant dialogue in this event"],
      "actions": [
        {
          "actor": "C1",
          "verb": "action verb",
          "object": "target object",
          "spatial_relation": "spatial context from the actor's perspective"
        }
      ]
    }
  ]
}

Important requirements:
1. Use relative spatial relations (above, below, left of, right of, in front of, behind, on top of, next to, inside, leaning against) for describing object positions
2. Describe positions from the actor's perspective, not the camera's perspective
3. Include actual spoken dialogue when audible, with accurate transcription
4. Identify all visible characters and assign them local IDs (C1, C2, etc.)
5. Capture state changes of objects (picked up, put down, opened, closed, moved, etc.)
6. Each event should represent a distinct action or occurrence

Return ONLY the JSON object with no additional text, explanations, or markdown formatting."""


PROMPT_CLIP_ANALYSIS_STRICT_JSON = """You must analyze this video clip and return ONLY a valid JSON object. No explanations, no markdown, just JSON.

Return a JSON object with these exact keys:
- clip_summary (string): 1-2 sentence summary
- scene_type (string): type of scene
- characters (array): list of character objects with local_character_id, role, name_or_description
- speaker_turns (array): list of speech objects with speaker_id, utterance, time_start, time_end
- events (array): list of event objects

Each event object must have:
- local_event_id (string)
- time_start (number)
- time_end (number)  
- summary (string)
- actors (array of strings)
- objects (array of objects with name and spatial_description)
- dialogue (array of strings)
- actions (array of action objects with actor, verb, object, spatial_relation)

Describe spatial relations from the actor's viewpoint. Include actual spoken dialogue.

Return ONLY valid JSON, starting with { and ending with }."""


# =============================================================================
# Event Summarization Prompts
# =============================================================================

PROMPT_EVENT_SUMMARIZATION = """Create a concise, information-dense summary of this event that captures the key details for memory retrieval and question answering.

Event information:
- Scene type: {scene_type}
- Event summary: {event_summary}
- Actors involved: {actors}
- Objects present: {objects}
- Dialogue: {dialogue}
- Actions: {actions}

Requirements:
1. Produce 1-2 short sentences maximum
2. Include the main behavior/action if present
3. Include key spatial configuration if relevant
4. Include important dialogue intent if speech occurred
5. Write in a way useful for later retrieval and question answering
6. Avoid redundant wording and filler phrases
7. Use character descriptions or IDs consistently

Summary:"""


PROMPT_MERGED_EVENT_SUMMARIZATION = """Create a unified summary covering multiple sequential events. These events were merged because they represent continuous activity.

Events to merge:
{events_info}

Requirements:
1. Create a single cohesive summary (2-3 sentences maximum)
2. Capture the overall progression of activity
3. Include key behaviors, spatial details, and dialogue intents
4. Maintain temporal order in the description
5. Avoid redundancy while preserving important details

Summary:"""


# =============================================================================
# Question Classification Prompts  
# =============================================================================

PROMPT_QUESTION_TYPING = """Classify the following question into one of these categories based on the type of reasoning and memory retrieval it requires:

Categories:
- Object_Tracking: Questions about where objects are located, how objects moved, or object state changes
- Person_Understanding: Questions about a person's characteristics, preferences, habits, occupation, relationships, or behavior patterns
- Temporal_Multihop: Questions requiring connecting events across different time periods
- Cross_Modal: Questions requiring integration of visual and audio information
- General: Questions that do not clearly fit the above categories

Question: {question}

Return a JSON object with:
- question_type: one of the category names above
- key_entities: list of important entities (people, objects) mentioned in the question
- reasoning_brief: one sentence explaining why this classification

Return ONLY the JSON object."""


# =============================================================================
# Retrieval and Search Prompts
# =============================================================================

PROMPT_SEARCH_QUERY_GENERATION = """Based on the question and the information retrieved so far, generate a search query to find relevant memories.

Question: {question}

Information retrieved so far:
{retrieved_info}

If the current information is sufficient to answer the question, return:
{{"action": "ANSWER", "content": "your answer here"}}

If more information is needed, return:
{{"action": "SEARCH", "query": "search query to find missing information"}}

The search query should be specific and target the missing information needed to answer the question.

Return ONLY the JSON object."""


PROMPT_RELEVANCE_SCORING = """Score the relevance of the following event summaries to the question.

Question: {question}

Events (ordered by time):
{events}

For each event, provide a relevance score from 0.0 to 1.0 based on how useful it is for answering the question.

Return a JSON array with scores in the same order as the events:
[score1, score2, score3, ...]

Return ONLY the JSON array."""


# =============================================================================
# Answer Generation Prompts
# =============================================================================

PROMPT_FINAL_ANSWER = """Based on the provided context, answer the question. If the answer cannot be determined from the context, say so explicitly.

Question: {question}

Retrieved Context (events ordered by time):
{context}

Requirements:
1. Answer based ONLY on the provided context
2. Be direct and concise
3. If the information is not in the context, state that clearly
4. Do not make assumptions beyond what the context supports

Answer:"""


PROMPT_ANSWER_WITH_REASONING = """Based on the provided context, answer the question. Provide your reasoning before the final answer.

Question: {question}

Retrieved Context (events ordered by time):
{context}

First explain your reasoning based on the context, then provide your final answer.

Format your response as:
Reasoning: [Your reasoning here]
Answer: [Your final answer here]"""


# =============================================================================
# Evaluation Prompts
# =============================================================================

PROMPT_VERIFY_ANSWER = """You are evaluating whether a predicted answer is correct given the question and ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Determine if the predicted answer is semantically consistent with the ground truth answer in the context of the question. The predicted answer does not need to match exactly - it should convey the same meaning.

Return only "Yes" if the predicted answer is correct, or "No" if it is incorrect."""


PROMPT_VERIFY_ANSWER_INFERENCE = """Determine whether the ground truth answer can be logically inferred from the predicted answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted}

Important:
- Do not require exact wording match
- The predicted answer is correct if the ground truth can be reasonably derived from it
- Consider semantic equivalence in the context of the question

Return only "Yes" if the ground truth can be inferred from the predicted answer, or "No" if it cannot."""


# =============================================================================
# Agent Control Prompts
# =============================================================================

PROMPT_AGENT_SYSTEM = """You are an intelligent assistant that answers questions by searching through a memory bank of video events. You have access to a memory retrieval system.

Question: {question}

At each step, you can either:
1. Search for more information by outputting: [SEARCH] your search query
2. Provide the final answer by outputting: [ANSWER] your answer

The search query should be specific and help find the information needed to answer the question.

Guidelines:
- Search for relevant events, character information, object locations, or dialogue
- You may need multiple searches to gather enough information
- When you have sufficient information, provide a direct answer
- If searching yields no results, try different search terms
- Character IDs may appear as <character_1>, <face_1>, <voice_1>, etc."""


PROMPT_AGENT_ACTION = """Based on the question and retrieved knowledge, decide whether to search for more information or provide an answer.

Question: {question}

Retrieved Knowledge:
{knowledge}

If the knowledge is sufficient, output:
[ANSWER] your direct answer to the question

If more information is needed, output:
[SEARCH] a specific search query to find missing information

Only include [ANSWER] or [SEARCH] followed by your content. Do not include both."""


PROMPT_AGENT_FINAL_ANSWER = """You must now provide a final answer based on all the information gathered.

Question: {question}

All Retrieved Information:
{information}

Provide your best answer based on the available information. If the information is insufficient, make a reasonable inference.

[ANSWER]"""


# =============================================================================
# Fallback Recovery Prompts
# =============================================================================

PROMPT_JSON_REPAIR = """The following text is a broken JSON response from a video analysis model. 
Fix it to produce valid JSON with these required keys:
- clip_summary (string)
- scene_type (string)
- characters (array)
- speaker_turns (array)
- events (array)

Each event must have: local_event_id, time_start, time_end, summary, actors, objects, dialogue, actions.

Broken response:
{broken_json}

Return ONLY the repaired valid JSON. Start with {{ and end with }}."""


PROMPT_SEMANTIC_DESCRIPTION = """Describe what happens in this video clip in 1-2 sentences.

Include these elements when visible or audible:
- Main actors (who is in the scene)
- Main actions (what they are doing)
- Key spatial relationships between people and objects (positions using: left of, right of, above, below, in front of, behind, next to, on top of)
- Main intent of any spoken dialogue (if clearly audible)

Write from the main actor's perspective.

Return ONLY the description text, no JSON, no formatting."""


PROMPT_NEIGHBOR_INTERPOLATION = """Given information about events before and after a video segment, write 1 sentence describing what most likely happens in the intermediate segment.

Time range: {time_start:.1f}s to {time_end:.1f}s

{neighbor_context}

Include who is present, what they are likely doing, and any relevant spatial context.
Return ONLY the sentence, no explanations."""


# =============================================================================
# Helper Functions
# =============================================================================

def format_events_for_context(events: list) -> str:
    """
    Format a list of events into a string for use in prompts.
    
    Args:
        events: List of event dictionaries or strings.
        
    Returns:
        Formatted string representation of events.
    """
    if not events:
        return "No events available."
    
    formatted = []
    for i, event in enumerate(events, 1):
        if isinstance(event, dict):
            time_str = ""
            if "time_start" in event:
                time_str = f"[{event['time_start']:.1f}s - {event.get('time_end', event['time_start']):.1f}s] "
            summary = event.get("summary_text", event.get("summary", str(event)))
            formatted.append(f"{i}. {time_str}{summary}")
        else:
            formatted.append(f"{i}. {event}")
    
    return "\n".join(formatted)


def format_dialogue_for_context(dialogue_snippets: list) -> str:
    """
    Format dialogue snippets into a readable string.
    
    Args:
        dialogue_snippets: List of dialogue strings.
        
    Returns:
        Formatted dialogue string.
    """
    if not dialogue_snippets:
        return ""
    return " | ".join(dialogue_snippets)


def format_actions_for_prompt(actions: list) -> str:
    """
    Format action list into a readable string.
    
    Args:
        actions: List of action dictionaries.
        
    Returns:
        Formatted actions string.
    """
    if not actions:
        return "None"
    
    formatted = []
    for action in actions:
        if isinstance(action, dict):
            actor = action.get("actor", "someone")
            verb = action.get("verb", "does something")
            obj = action.get("object", "")
            spatial = action.get("spatial_relation", "")
            
            action_str = f"{actor} {verb}"
            if obj:
                action_str += f" {obj}"
            if spatial:
                action_str += f" ({spatial})"
            formatted.append(action_str)
        else:
            formatted.append(str(action))
    
    return "; ".join(formatted)


if __name__ == "__main__":
    # Test prompt formatting
    test_events = [
        {"time_start": 0.0, "time_end": 5.0, "summary_text": "A person enters the room"},
        {"time_start": 5.0, "time_end": 10.0, "summary_text": "They pick up a book from the table"},
    ]
    print("Formatted events:")
    print(format_events_for_context(test_events))
