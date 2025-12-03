# README

Adaptive Multi-Channel Relational Memory Graph (AMR-Graph)
Online Streaming Version. 

This document specifies the full implementation design for the **Adaptive Multi-Channel Relational Memory Graph (AMR-Graph)** used for multimodal long-term memory reasoning over long videos.
All modules must support **online ingestion**, meaning the graph must grow dynamically as new video clips arrive.
No offline full-video processing is allowed.

All code should be written in **Python 3.10**, compatible with **CUDA 12.2** and a single **NVIDIA RTX 4090**.
External API requests must use **OpenRouter** (e.g., `google/gemini-2.5-pro`, `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-5`, `openai/text-embedding-3-large`).

---

# Part I — Memory Storage (Online Dynamic Construction)

## 1. Overview

The Memory Storage module converts streaming video clips into event-level memory nodes and incrementally updates a multi-channel relational graph.
Processing is strictly **online**, i.e., each clip is processed immediately when it arrives.

The pipeline consists of:

1. Video clip ingestion (configurable window, default 10 seconds)
2. Multimodal event extraction via LLM (Gemini through OpenRouter)
3. Event node construction and embedding generation
4. Optional adaptive event merging (local optimization, modular & switchable)
5. Dynamic AMR-Graph update with incremental edge construction
6. Incremental indexing (BM25, FAISS, person/object indices)

---

## 2. Video Clip Processing Pipeline

### 2.1 Clip segmentation

The system receives a sequence of clips:

```text
(video_id, clip_id, t_start, t_end, clip_video_segment)
```

Clips are processed in **timestamp order**.

* Use a **target clip length** `TARGET_CLIP_SEC` (default: `10.0` seconds), configurable via a global config or CLI flag.

* When segmenting a long video stream:

  * Initialize `t = 0`.
  * Repeatedly create clips `[t, t + TARGET_CLIP_SEC]`.
  * If the remaining duration `< TARGET_CLIP_SEC`, create a **short final clip** `[t, t_final]` without padding.
  * The implementation must **not assume all clips are exactly 10 seconds**; all downstream logic must handle variable clip lengths.

* For externally pre-segmented clips, the system must **respect input timestamps** and treat `t_start, t_end` as ground truth, even if the length differs from `TARGET_CLIP_SEC`.

This ensures robust behavior for arbitrary video lengths and partial segments.

---

### 2.2 Multimodal analysis (LLM via OpenRouter)

Each clip is sent to `google/gemini-2.5-pro` (or another compatible multimodal LLM) via OpenRouter, using **video + audio input**.

The LLM must return a **strictly valid JSON** object with the following structure:

```json
{
  "clip_summary": "...",
  "scene_type": "...",
  "characters": [
    {
      "local_character_id": "C1",
      "role": "speaker/observer/other",
      "name_or_description": "girl in black hoodie"
    }
  ],
  "speaker_turns": [
    {
      "speaker_id": "S1",
      "utterance": "...",
      "time_start": 3.5,
      "time_end": 5.2
    }
  ],
  "events": [
    {
      "local_event_id": "E1",
      "time_start": 2.0,
      "time_end": 8.5,
      "summary": "...",
      "actors": ["C1", "C2"],
      "objects": [
        {
          "name": "phone",
          "spatial_description": "to the left of the mug on the table"
        }
      ],
      "dialogue": ["...", "..."],
      "actions": [
        {
          "actor": "C1",
          "verb": "pick up",
          "object": "phone",
          "spatial_relation": "from the table in front of her"
        }
      ]
    }
  ]
}
```

#### Prompt requirements for spatial relations & perspective

The system’s prompt to the LLM must enforce:

1. **Relative spatial relations** for objects sharing the same surface or local area, using explicit vocabulary such as:

   * `above / below`
   * `left of / right of`
   * `in front of / behind`
   * `on top of / underneath`
   * `next to / beside`
   * `inside / outside`
   * `leaning against`

   Example descriptions requested from the model:

   * `"Jacket is hanging ABOVE the hat on the coat rack"`
   * `"Phone is placed TO THE LEFT OF the mug on the table"`
   * `"Backpack is leaning AGAINST the bed frame, NEXT TO the pillow"`
   * `"Keys are ON TOP OF the book, which is on the desk"`

2. **Actor-centric perspective** (PERSPECTIVE requirement):

   * Directions (left/right, front/behind) must be described from the **actor’s viewpoint**, not from a detached camera perspective.
   * Example: `"From the man's perspective, the laptop is in front of him and the phone is to his right on the desk."`

3. **Strict JSON output**:

   * The prompt must explicitly instruct:

     * “Return **only** a single JSON object, with no explanations, comments, or extra text.”
     * “No trailing commas, no additional keys beyond the schema.”
     * “Do not include markdown fences, no natural language outside the JSON.”

Given the fragility of JSON parsing in long video pipelines, the code must implement:

* A **strict JSON parser** with:

  * Basic sanitization (trimming whitespace, removing markdown fences if present).
  * A fallback retry strategy with a stricter prompt if parsing fails the first time.
* Validation of required keys (`clip_summary`, `events`, etc.) before proceeding.

---

## 3. Event Node Construction

For each event returned by the LLM, construct an `AdaptiveEventNode`.

### 3.1 AdaptiveEventNode schema

```python
class AdaptiveEventNode:
    node_id: int           # global sequential ID, starting from 0
    video_id: str
    clip_ids: list[str]

    time_start: float
    time_end: float

    summary_text: str      # short, task-oriented LLM summary
    dialogue_snippets: list[str]

    persons: list[str]     # actors + speaker IDs + mentioned names
    objects: list[str]     # object names from LLM event parsing
    scene_type: str

    actions: list[dict]    # [{actor, verb, object, spatial_relation}]

    text_embedding: np.ndarray
```

#### Node ID assignment

* Maintain a global counter `next_node_id` starting at `0`.

* Whenever a new event node is created (before any optional merging):

  ```python
  node_id = next_node_id
  next_node_id += 1
  ```

* `node_id` must remain stable and **monotonically increasing** across the entire run, ensuring that graph nodes are easily interpretable and human-debuggable.

---

### 3.2 LLM-driven extraction details

All semantic fields are driven by LLM outputs rather than classical NLP pipelines.

**Persons**

* Combine:

  * `actors` from the event-level LLM output.
  * Speaker IDs from `speaker_turns`.
  * Proper names and role descriptions explicitly listed in `events[].summary`, `events[].dialogue`, and `clip_summary` (using an additional lightweight LLM call if needed).

* A secondary LLM call may be used to normalize person descriptors to short stable strings, e.g.:

  * `"girl in black hoodie"` → `"girl_black_hoodie"`
  * `"older man with glasses"` → `"older_man_glasses"`

**Objects**

* Directly use the `objects` field from the event JSON:

  * Each object should include `name` and an optional `spatial_description`.
  * Implement a normalization step (e.g., lowercasing, singularization) when building indices, without altering the original text.

**Actions**

* Actions are **not** derived by dependency parsing; instead:

  * The event JSON already includes `actions`:

    ```json
    {
      "actor": "C1",
      "verb": "pick up",
      "object": "phone",
      "spatial_relation": "from the table in front of her"
    }
    ```

  * If needed, a second LLM pass over `(summary, dialogue, objects)` can refine or complete the `actions` list (e.g., fill missing spatial relations, normalize verbs).

* The implementation should treat actions as LLM-specified tuples `{actor, verb, object, spatial_relation}`.

---

### 3.3 Node summary generation (LLM-synthesized, not concatenated)

Each node must have a **short, clean, task-oriented** `summary_text`, used as the primary index text.

Requirements:

* **Do NOT** construct summaries by directly concatenating:

  * All dialogue lines
  * All object lists
  * Raw descriptions

* Instead, make a **dedicated LLM call** to synthesize a compact summary from structured fields:

  Input to the summarization LLM:

  * Event-level fields:

    * `clip_summary`
    * `event.summary`
    * `event.dialogue`
    * `event.actors`
    * `event.objects` (with spatial descriptions)
    * `event.actions`
    * `scene_type`

  * Explicit instructions:

    * “Produce a single sentence or at most 2 short sentences.”
    * “The summary should be **information-dense** and include, when available:
      (1) main behavior/action,
      (2) key spatial configuration,
      (3) important dialogue intent.”
    * “Write in a way that is useful for later video question answering and memory retrieval.”
    * “Avoid redundant wording and filler phrases.”

Example desired style:

> “From the woman’s perspective, she picks up the phone to the left of her mug on the table and tells her friend that she is leaving soon.”

This ensures each node’s `summary_text` is **compact**, **query-friendly**, and captures **behavior + space + dialogue** in a single dense representation.

**Text embeddings**

* Use `openai/text-embedding-3-large` via OpenRouter on `summary_text` (optionally plus a lightweight combination of key fields).
* Store the resulting vector in `text_embedding`.

---

## 4. Adaptive Event Merging (Local Compression, Modular & Switchable)

To prevent uncontrolled memory growth, neighboring event nodes may be merged if they represent continuous actions.

Merging criteria:

1. Same `video_id`,
2. Time gap ≤ `MERGE_TIME_GAP_THRESHOLD`,
3. `cos_sim(A.embedding, B.embedding) > MERGE_SIM_THRESHOLD`,
4. Overlap in `persons` or `scene_type`.

**Local comparison**: only compare newly created events with the most recent events for that video.

### 4.1 Modular design

* Implement merging as a **separate module** or function (e.g., `merge_candidate_nodes(...)`).

* Control merging with a configuration flag, e.g.:

  ```yaml
  MERGE_EVENTS_ENABLED: true
  ```

* When `MERGE_EVENTS_ENABLED` is `false`:

  * The system creates one node per event and **skips merging** entirely.
  * All other logic (edges, indices) must still function correctly.

### 4.2 Post-merge updates

When two or more nodes are merged into a new node:

* Recompute `summary_text` via LLM:

  * Provide the LLM with the original summaries, dialogues, and actions of all merged nodes.
  * Request a **single compact summary** covering the merged time span, again emphasizing **behavior + space + dialogue**.

* Recompute `text_embedding` on the new `summary_text`.

* Union and deduplicate `persons`, `objects`, and `actions`.

* Preserve `time_start` as the earliest and `time_end` as the latest among merged nodes.

* The resulting merged node receives a **new `node_id`** via the global sequential allocator (do not reuse IDs).

---

## 5. AMR-Graph Structure

### 5.1 Nodes

Nodes are all `AdaptiveEventNode` instances with integer `node_id`s starting from 0.

### 5.2 Edge channels

The graph maintains several relation channels:

```python
edges = {
    "temporal": {src_id: [(dst_id, payload)]},
    "entity_jump": {src_id: [(dst_id, payload)]},
    "object_lifecycle": {src_id: [(dst_id, payload)]},
    "cross_modal": {src_id: [(dst_id, payload)]}  # optional
}
```

* `payload` may contain metadata such as time gap, shared entities, or action type.

---

## 6. Dynamic Graph Update (Online)

When a new batch of event nodes is produced from a clip:

### 6.1 Append new nodes

Add each node to:

* `global_event_dict[node_id] = node`
* `video_index[video_id].append(node_id)`

`node_id` is always assigned from the global incremental counter.

---

### 6.2 Temporal edges

For each new node:

* Connect from the last event of the same video, if any:

```text
temporal: prev_node_id → new_node_id
```

* Store the time gap in the edge payload, if needed.

---

### 6.3 Entity jump edges

Maintain an index:

```python
person_index: dict[str, list[int]]  # person_id -> list of node_ids
```

For each person `p` in the new event:

* Look up the most recent node `prev_node_id` in `person_index[p]`.
* If `time_gap` exceeds a threshold (indicating a non-local reappearance), add:

```text
entity_jump: prev_node_id → new_node_id
```

* Append `new_node_id` to `person_index[p]`.

---

### 6.4 Object lifecycle edges

Maintain an index:

```python
object_index: dict[str, list[int]]  # object_name -> list of node_ids
```

For each action `(actor, verb, object, spatial_relation)`:

* Look up the previous node `prev_node_id` that contains this `object`.
* If the verb suggests a state transition (e.g., “pick up”, “put down”, “open”, “close”), add:

```text
object_lifecycle: prev_node_id → new_node_id
```

* Append `new_node_id` to `object_index[object]`.

---

### 6.5 Optional cross-modal edges

If the system later adds online face/voice identity tracking:

* Create `cross_modal` edges between nodes that link:

  * voice-only events and
  * face-visible events

for the same real-world person.

---

## 7. Indexing

### 7.1 BM25 Index (Sparse)

Index the following fields for keyword-based retrieval:

* `summary_text`
* `dialogue_snippets`
* `persons`
* `objects`

The BM25 index must support **incremental updates** as new nodes are added.

---

### 7.2 FAISS (Dense embedding) index

Store `text_embedding` for each node in a FAISS index to support dense retrieval.

* Use `add_with_ids` with `node_id` as the index ID to synchronize FAISS with graph IDs.
* Support incremental add as new nodes are created.

---

### 7.3 Person/Object indices

Maintain dictionary-based inverted indices, updated online:

```python
person_index[person] -> list[node_id]
object_index[obj] -> list[node_id]
```

These indices are used both for edge construction and for targeted retrieval (e.g., all events involving a specific person or object).

---

# Part II — Memory Retrieval

**Overall strategy:** Anchor → Navigate → Verify

Memory retrieval is performed entirely over:

* the AMR-Graph (multi-channel relations),
* the dense / sparse indices, and
* a three-stage pipeline.

---

## 1. Question Typing (LLM-driven)

Instead of rule-based or regex classification, question typing is handled by a **LLM** (e.g., `gpt-4o`) via OpenRouter.

The LLM receives:

* The user query `Q`.
* A short description of available task types.

It must output a **strict JSON** label from:

* `Object_Tracking`
* `Person_Understanding`
* `Temporal_Multihop`
* `Cross_Modal`
* `General`

Example JSON:

```json
{
  "question_type": "Object_Tracking",
  "reason": "The question asks where an object moves over time."
}
```

The implementation uses `question_type` to decide channel priorities in the navigation stage; the `reason` is optional and can be ignored by the core logic.

---

## 2. Stage 1 — Anchor Retrieval

Given query `Q`:

### 2.1 Sparse retrieval (BM25)

Retrieve top-N events by keyword matching over the BM25 index.

### 2.2 Dense retrieval (FAISS)

* Generate a query embedding from `Q` using `text-embedding-3-large`.
* Retrieve top-M similar events from FAISS.

### 2.3 Fusion

Use **Reciprocal Rank Fusion (RRF)** to combine sparse and dense results, then select top-K anchors (e.g., K=5–10).

These anchors initialize the graph traversal.

---

## 3. Stage 2 — Graph Navigation with Channel Switching

Navigation involves exploring the AMR-Graph from anchors using channel priority rules derived from `question_type`.

### 3.1 Channel priority

Examples:

**Object_Tracking**

1. `object_lifecycle`
2. `temporal`
3. `entity_jump`

**Person_Understanding**

1. `entity_jump`
2. `temporal`

**Temporal_Multihop**

1. `temporal`
2. `entity_jump`

**Cross_Modal**

1. `cross_modal`
2. `temporal`

**General**

* Balanced mix of `temporal` and `entity_jump`, with occasional `object_lifecycle` edges.

---

### 3.2 Multi-channel BFS / Beam Search

Maintain a frontier of nodes with a configurable beam width (e.g., 20).

At each expansion step:

1. For each node in the current frontier:

   * Explore outgoing edges in the order of channel priority.
   * For every candidate neighbor, compute a relevance score that combines:

     * **Channel weight** (depends on question type).
     * **Textual similarity** between neighbor `summary_text` and query `Q` (can use inner product in embedding space).

2. Select top-scoring neighbors to form the next frontier.

Stop when:

* Maximum hop count (e.g., 3–5) is reached, or
* Maximum visited nodes limit is hit, or
* The frontier scores converge.

The output is a **candidate set of events** or **candidate paths** related to the query.

---

## 4. Stage 3 — Verification

This step filters noise and selects the most relevant event groups before final LLM answering.

### 4.1 Grouping candidates

Group candidates by:

* Traversal path, or
* Time window clustering (e.g., events within a continuous time segment).

### 4.2 LLM relevance scoring

For each candidate group:

* Compose a concise prompt containing:

  * A list of event `summary_text`s (ordered by time).
  * The original question `Q`.

* Send this to a compact LLM (e.g., `gpt-5`) and request:

  * A relevance score `0–1`.
  * Optionally, a short explanation (not required by the core pipeline).

Only top-scoring groups are kept.

---

### 4.3 Final context assembly

Collect events from top groups, order them by `time_start`, and compact them into a textual block:

* Focus on `summary_text` as the backbone.
* Optionally include key dialogue snippets where they are critical to answering the question.
* Keep the context concise to avoid overwhelming the final LLM.

---

## 5. Final Answer Generation

Use a strong model (e.g., `gpt-4o` or `google/gemini-2.5-pro`) via OpenRouter.

The input prompt to the final LLM should contain:

* The compacted event list (with time order and short summaries).
* The user question `Q`.
* Instructions to:

  * Answer **strictly based on the provided context**.
  * Explicitly state if the answer cannot be found in the context.

The LLM then produces the final answer for the user.

---

## 6. Implementation Notes (JSON Robustness & Safety)

* **JSON parsing** is a critical failure point:

  * Always instruct LLMs to output **only JSON**, with no additional explanations.
  * Implement strict validation for required keys, types, and basic schema structure.
  * Add retry logic with a more constrained prompt when parsing fails.

* All critical LLM calls that must return JSON (clip analysis, question typing, node summarization) should:

  * Use deterministic settings where possible (e.g., `temperature` close to 0 when structure is more important than creativity).
  * Limit maximum tokens to reduce the risk of extraneous content.

* All components (clip processing, node construction, merging, indexing, retrieval) must be designed to handle:

  * Variable clip durations,
  * Optional disabled merging,
  * Incremental online updates without full graph recomputation.
