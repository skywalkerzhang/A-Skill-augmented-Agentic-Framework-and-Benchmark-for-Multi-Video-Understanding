# SAMA

Code for **SAMA: A Skill-augmented Agentic Framework for Multi-Video Understanding**.

## Install

Python 3.10 and a CUDA GPU are recommended.

```bash
git clone <repo-url>
cd Multivideo_Agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

Edit `.env` and set:

- `API_MODEL_NAME`, `API_BASE_URL`, `API_KEY`: text-only planner
- `API_MODEL_NAME_VLM`: local VLM path or Hugging Face model ID
- For a remote VLM, also set `VLM_USE_REMOTE=1`, `API_BASE_URL_VLM`, and `API_KEY_VLM`
- `BGE_M3_PATH`: local BGE-M3 path or `BAAI/bge-m3`

## Data

The input is a JSON list:

```json
[
  {
    "question": "Which candidate matches the reference video?",
    "answer": "A",
    "video_paths": [
      "/path/to/reference.mp4",
      "/path/to/A.mp4",
      "/path/to/B.mp4",
      "/path/to/C.mp4",
      "/path/to/D.mp4"
    ]
  }
]
```

Convert a JSONL file when needed:

```bash
python -m multivideo_agent.convert_jsonl_to_json \
  input.jsonl qa.json /path/to/dataset
```

## Run

```bash
bash scripts/run_eval.sh \
  qa.json \
  /path/to/dataset \
  outputs/sama.json
```

Show all options:

```bash
python -m multivideo_agent.eval_multivideo --help
```
