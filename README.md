# AMR-Graph: Adaptive Multi-Channel Relational Memory Graph Pipeline

A memory graph structure for long-term video reasoning, evaluated on the M3-Bench benchmark.

## Hardware Requirements

- **OS**: Linux
- **GPU**: NVIDIA GeForce RTX 4090
- **CUDA**: 12.2
- **Python**: 3.10

## Environment Setup

1. Create and activate conda environment:
```bash
conda create -n new python=3.10
conda activate new
```

2. Install system dependencies (ffmpeg for video processing):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Or if using conda
conda install -c conda-forge ffmpeg
```

3. Install project dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API credentials:
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Usage

The pipeline provides three main commands: `build`, `evaluate`, and `query`.

### Quick Start

To run the complete pipeline on a video from start to finish, including building the memory graph, retrieving context, answering questions, and evaluating results, use the `evaluate` command:

```bash
python main.py evaluate \
  --video_path /home/alex/thesis/data/video/robot/bedroom_01.mp4 \
  --log-file /home/alex/thesis/data/logs/bedroom_01_eval.log
```

This single command will process the entire video, build the memory graph, answer all questions for that video, and save comprehensive evaluation results.

### Build Memory Graph

Build a memory graph from a video file without running evaluation.

```bash
python main.py build \
  --video_path /home/alex/thesis/data/video/robot/bedroom_01.mp4 \
  --output /home/alex/thesis/data/memory_graphs/bedroom_01_graph.json \
  --log-file /home/alex/thesis/data/logs/bedroom_01_build.log
```

**Flags:**
- `--video_path`: Path to the input video file (required)
- `--output`, `-o`: Path to save the generated memory graph JSON (optional, default: `data/memory_graphs/<video_name>_graph.json`)
- `--max-clips`: Maximum number of video clips to process (optional, processes entire video by default)
- `--clip-duration`: Target duration for each video clip in seconds (optional, uses config default)
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--log-file`: Path to save log file (optional, prints to console if not specified)

### Evaluate on QA Dataset

Run complete pipeline: build memory graph (or load existing) and evaluate on M3-Bench QA.

```bash
python main.py evaluate \
  --video_path /home/alex/thesis/data/video/robot/bedroom_01.mp4 \
  --log-file /home/alex/thesis/data/logs/bedroom_01_eval.log
```

This command will:
1. Build the memory graph from the video (or use existing if `--graph-path` is provided)
2. Load corresponding QA pairs from the annotation file
3. Retrieve relevant context from the memory graph for each question
4. Generate answers using the LLM
5. Evaluate answers against ground truth
6. Save results, summary, timing report, and accuracy table

**Flags:**
- `--video_path`: Path to the input video file (required unless using `--graph-path`)
- `--annotations`, `-a`: Path to M3-Bench annotation JSON file (default: `data/annotations/robot.json`)
- `--video-name`: Video identifier for filtering QA pairs (default: extracted from video filename)
- `--output-dir`, `-o`: Directory to save evaluation results (default: `data/results`)
- `--graph-path`: Load existing memory graph instead of building from video (optional)
- `--max-clips`: Maximum number of video clips to process during graph building (optional)
- `--clip-duration`: Target duration for each video clip in seconds (optional)
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--log-file`: Path to save log file (optional)

**Output Files** (saved in `--output-dir`):
- `<video_name>_results.jsonl`: Detailed results for each question
- `<video_name>_summary.json`: Summary statistics including accuracy by question type
- `<video_name>_timing.txt`: Timing breakdown for each pipeline stage
- `<video_name>_acctable.txt`: Formatted accuracy table by question type

### Query Memory Graph

Run a single query against a pre-built memory graph.

```bash
python main.py query \
  /home/alex/thesis/data/memory_graphs/bedroom_01_graph.json \
  "What coat rack should Emma's coat be placed on?" \
  --log-file /home/alex/thesis/data/logs/query.log
```

**Arguments:**
- `graph_path`: Path to saved memory graph JSON file (required, positional)
- `question`: Question to answer (required, positional)

**Flags:**
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--log-file`: Path to save log file (optional)

## Example Workflows

### Evaluate a Single Video from Scratch

```bash
python main.py evaluate \
  --video_path /home/alex/thesis/data/video/robot/bedroom_01.mp4 \
  --log-file /home/alex/thesis/data/logs/bedroom_01_eval.log
```

### Evaluate Using Pre-built Graph

```bash
python main.py evaluate \
  --graph-path /home/alex/thesis/data/memory_graphs/bedroom_01_graph.json \
  --video-name bedroom_01 \
  --log-file /home/alex/thesis/data/logs/bedroom_01_eval.log
```

### Process Only First 50 Clips of a Video

```bash
python main.py build \
  --video_path /home/alex/thesis/data/video/robot/bedroom_01.mp4 \
  --max-clips 50 \
  --output /home/alex/thesis/data/memory_graphs/bedroom_01_partial_graph.json
```

### Test Retrieval on Custom Question

```bash
python main.py query \
  /home/alex/thesis/data/memory_graphs/bedroom_01_graph.json \
  "Is Emma's tidying up habit good?"
```

## Project Structure

```
/home/alex/thesis/
├── main.py                 # CLI entry point
├── amr_graph.py           # Memory graph structure
├── video_processor.py     # Video processing and clip extraction
├── event_node.py          # Event node definitions
├── indexing.py            # Vector indexing and embeddings
├── retrieval.py           # Retrieval pipeline
├── evaluation.py          # Answer evaluation and result tracking
├── prompts.py             # LLM prompts
├── api_utils.py           # API interaction utilities
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── configs/
│   └── api_config.json    # Model configurations
├── data/
│   ├── video/robot/       # M3-Bench robot videos
│   ├── annotations/       # QA annotation files
│   ├── memory_graphs/     # Generated memory graphs
│   ├── results/           # Evaluation results
│   └── logs/              # Log files
└── .env                   # API keys and base URLs
```

## Evaluation Metrics

The evaluation results include:

- **Overall Accuracy**: Percentage of correctly answered questions
- **Accuracy by Question Type**: Performance breakdown for 5 M3-Bench question types:
  - Person Understanding
  - Cross-Modal Reasoning
  - Multi-Hop Reasoning
  - Multi-Detail Reasoning
  - Temporal Reasoning
- **Timing Breakdown**: Time spent in each pipeline stage:
  - Memory graph building
  - Index building
  - Retrieval per question
  - Evaluation per question
  - Total evaluation time

## Notes

- Questions can have multiple types. The evaluation correctly counts each type separately.
- The memory graph is built incrementally as video clips are processed (streaming approach).
- All results, timing reports, and accuracy tables are saved to the results directory for reproducibility.
- Log files provide detailed information about the pipeline execution for debugging.
- Embeddings are generated via OpenAI API (not locally), reducing local compute requirements.
- Video processing uses ffmpeg via subprocess calls (ffmpeg must be installed on system).
- FAISS-GPU is recommended for faster vector search, but will fallback to CPU if GPU is unavailable.

## License

This project is for research purposes related to long-term video reasoning and memory structures.
