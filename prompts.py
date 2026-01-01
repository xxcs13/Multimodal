"""
Prompt templates for AMR-Graph pipeline.

This module contains all prompt templates used for video analysis,
event summarization, question classification, retrieval, and answer generation.
All prompts are designed to be general and not task-specific.
"""

# =============================================================================
# Video Clip Analysis Prompts
# =============================================================================

PROMPT_CLIP_ANALYSIS = """You are a multimodal video analysis system.

=================================
CRITICAL TIME CONTEXT
=================================

You are analyzing:
- Clip ID: {clip_id}
- Clip Duration: {clip_duration_sec:.1f} seconds
- Global Start Time: {clip_global_start_sec:.1f}s (in the full video)

ALL TIMESTAMPS you output MUST be CLIP-LOCAL (ranging from 0.0 to {clip_duration_sec:.1f}).
Do NOT use global video timestamps.

=================================
HARD REQUIREMENTS
=================================

COVERAGE REQUIREMENTS:
1. Your events MUST collectively cover the ENTIRE clip duration [0.0, {clip_duration_sec:.1f}].
2. Total time gaps between events must be <= 1.0 second.
3. If a segment has no significant action, create a "background_continuation" event to fill the gap.
4. Events should be consecutive and non-overlapping.
5. Typical event duration: 2-10 seconds. Adjust based on content.

DIALOGUE REQUIREMENTS:
1. If ANY speech is audible, speaker_turns MUST NOT be empty.
2. Transcribe ALL dialogue VERBATIM (word-for-word, no summarizing).
3. For speakers visible on screen: use their character ID (C1, C2, etc.).
4. For speakers NOT visible on screen: use OFFSCREEN_1, OFFSCREEN_2, etc.
5. Do NOT use descriptive sentences like "The woman talks about..." - transcribe actual words.

=================================
JSON SCHEMA SPECIFICATION
=================================

You must output ONE valid JSON object with these fields:

clip_summary (string)
- 1-2 sentence summary of main activities in this clip.

scene_type (string)
- Environment type (indoor_living, outdoor_street, office, kitchen, bedroom, etc.).

characters (array)
- Only list VISIBLE characters on screen.
- Fields:
  - local_character_id (string): C1, C2, etc.
  - role (string): speaker, observer, or other.
  - name_or_description (string): Name if mentioned in audio, otherwise appearance description.

speaker_turns (array)
- Each entry = one continuous spoken segment.
- Fields:
  - speaker_id (string): Character ID (C1, C2) or OFFSCREEN_1, OFFSCREEN_2 for off-screen speakers.
  - utterance (string): EXACT verbatim dialogue (no paraphrasing).
  - time_start (number): Clip-local start time (0.0 to {clip_duration_sec:.1f}).
  - time_end (number): Clip-local end time.

events (array)
- Each entry = a distinct action, interaction, or occurrence.
- Fields:
  - local_event_id (string): E1, E2, etc.
  - time_start (number): Clip-local start time.
  - time_end (number): Clip-local end time.
  - summary (string): What happened in this event.
  - actors (array of strings): Character IDs involved (can be empty for pure environmental events).
  - objects (array):
      - name (string): Object name.
      - spatial_description (string): Position using: above, below, left of, right of, in front of, behind, on top of, next to, inside, leaning against.
  - dialogue (array of strings): Dialogue during this event (if any).
  - actions (array):
      - actor (string): Character ID.
      - verb (string): Action verb (pick up, open, point at, walk, sit, etc.).
      - object (string): Target object (if applicable).
      - spatial_relation (string): Spatial relation FROM THE ACTOR'S PERSPECTIVE.

=================================
SPATIAL CONSTRAINTS
=================================

1. Describe positions from the ACTOR'S perspective, NOT the camera's.
2. Use ONLY relative spatial terms.
3. Capture object state changes (picked up, put down, opened, closed, moved).

=================================
OUTPUT FORMAT
=================================

- Output ONLY valid JSON.
- Start with {{ and end with }}.
- No explanations, comments, markdown, or extra text.
"""


PROMPT_CLIP_ANALYSIS_STRICT_JSON = """Analyze video clip and return ONLY valid JSON. No explanations, no markdown, just JSON.


CRITICAL CONSTRAINTS:
- ALL timestamps MUST be CLIP-LOCAL (0.0 to clip duration, NOT global timestamps)
- Events MUST cover the ENTIRE clip duration with minimal gaps
- Transcribe ALL dialogue VERBATIM - no paraphrasing
- For off-screen speakers, use OFFSCREEN_1, OFFSCREEN_2, etc.

Required JSON structure:
{{
  "clip_summary": "1-2 sentence summary",
  "scene_type": "environment type",
  "characters": [
    {{"local_character_id": "C1", "role": "speaker/observer", "name_or_description": "..."}}  
  ],
  "speaker_turns": [
    {{"speaker_id": "C1", "utterance": "exact words", "time_start": 0.0, "time_end": 3.0}}
  ],
  "events": [
    {{
      "local_event_id": "E1",
      "time_start": 0.0,
      "time_end": 5.0,
      "summary": "what happened",
      "actors": ["C1"],
      "objects": [{{"name": "...", "spatial_description": "..."}}],
      "dialogue": ["utterances during this event"],
      "actions": [{{"actor": "C1", "verb": "...", "object": "...", "spatial_relation": "..."}}]
    }}
  ]
}}

Return ONLY valid JSON, starting with {{ and ending with }}."""


# =============================================================================
# Event Summarization Prompts
# =============================================================================

PROMPT_EVENT_SUMMARIZATION = """Create a concise summary for memory retrieval and question answering.

Event context:
- Time range: {time_start:.1f}s - {time_end:.1f}s
- Scene type: {scene_type}
- Event: {event_summary}
- Actors: {actors}
- Objects: {objects}
- Dialogue: {dialogue}
- Actions: {actions}

Requirements:
1. Produce 1-2 short sentences maximum.
2. Include the main action or behavior.
3. Include key spatial details if relevant.
4. Include dialogue content (not just "someone spoke") if speech occurred.
5. Use character descriptions consistently.
6. Avoid filler phrases.

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


PROMPT_INTEGRATE_CONTEXT = """You are given a list of event descriptions that are relevant to answering a question. Your task is to integrate and summarize these events into a coherent narrative that provides context for answering the question.

Question: {question}

Events (ordered by time):
{events}

Requirements:
1. Synthesize the events into a natural, flowing narrative
2. Preserve all important details relevant to the question
3. Maintain temporal order and relationships between events
4. Remove redundant information while keeping key facts
5. Be concise but comprehensive

Integrated Context:"""


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

