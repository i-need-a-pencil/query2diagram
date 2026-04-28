# Query2Diagram: Answering Developer Queries with UML Diagrams

Research code + data for **“Query2Diagram: Answering Developer Queries with UML Diagrams”** (arXiv:2604.23816).

**What it does**
Given:

- a code snippet (usually one file), and
- a natural-language developer query,

Query2Diagram generates a **focused UML-like diagram** as a **structured JSON graph** (`nodes`, `edges`, `packages`). The graph can be validated and converted to **PlantUML** (or Mermaid) for rendering.

**Paper:** [Query2Diagram: Answering Developer Queries with UML Diagrams](https://arxiv.org/abs/2604.23816)\
**HuggingFace Papers:** [Query2Diagram: Answering Developer Queries with UML Diagrams](https://huggingface.co/papers/2604.23816)\
**Diagram annotation interface:** [arboreal](https://github.com/i-need-a-pencil/arboreal)

---

## Quickstart (Docker-only)

### 0) Prereqs

- Docker + NVIDIA Container Toolkit (for `--gpus all`)

### 1) Build the image

From repo root:

```bash
docker build -t query2diagram .
```

### 2) Start a container (recommended: mount repo + HF cache)

This keeps your outputs (in `./datasets`) and model downloads cached across runs:

```bash
docker run --gpus all --ipc=host -it --rm \
  -p 8000:8000 \
  -v "$PWD":/q2d \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  query2diagram bash
```

You are now **inside the container** at `/q2d`.

---

## Run inference

Query2Diagram expects an **OpenAI-compatible** endpoint at `http://127.0.0.1:8000/v1`.
This repo uses **vLLM**.

### A) Fine-tuned inference (LoRA) — recommended

1. Start vLLM with LoRA enabled:

```bash
bash scripts/serve.sh
```

This serves:

- base model: `Qwen2.5-Coder-14B-Instruct-bnb-4bit`
- LoRA adapter: `./datasets/finetuned_model`
- model name exposed to clients: `finetuned_model`

2. In a **second terminal**, run generation:

```bash
# on host:
docker ps  # find container id/name
docker exec -it <container> bash -lc "python3.12 q2d/generation/generate_diagrams_local_finetuned.py"
```

Outputs:

- reads: `./datasets/test.json`
- writes: `./datasets/finetuned_model.json`

### B) Baseline inference (no LoRA)

If you don’t have `./datasets/finetuned_model` yet:

1. Start vLLM **without** LoRA (run directly instead of `scripts/serve.sh`):

```bash
vllm serve \
  --gpu-memory-utilization 0.5 \
  --max-model-len 17000 \
  --trust_remote_code \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype float16 \
  Qwen2.5-Coder-14B-Instruct-bnb-4bit
```

2. Generate:

```bash
python3.12 q2d/generation/generate_diagrams_local.py
```

Outputs:

- reads: `./datasets/test.json`
- writes: `./datasets/Qwen_base.json`

---

## Training (LoRA SFT)

Training uses **LLaMA-Factory** and writes the adapter to `./datasets/finetuned_model`.

### 1) Convert dataset → Alpaca JSON

```bash
python3.12 q2d/training/to_alpaca.py
```

Writes:

- `./datasets/training/diagrams_alpaca.json`
- `./datasets/training/diagrams_alpaca_eval.json`

### 2) Run LoRA SFT

```bash
llamafactory-cli train q2d/training/llama3_lora_sft.yaml
```

Output (LoRA adapter directory):

- `./datasets/finetuned_model`

> Note: `scripts/train.sh` points to `./training/llama3_lora_sft.yaml` (a path that doesn’t exist in this repo). Prefer the command above.

### 3) Serve + infer

```bash
bash scripts/serve.sh
python3.12 q2d/generation/generate_diagrams_local_finetuned.py
```

---

## Collect data (repo mining pipeline)

The mining pipeline lives in `q2d/datasets/*`.

### 1) Create a repo list

Create `./datasets/top_150.csv` with a `repo` column containing git URLs.

### 2) Download repos

```bash
python3.12 -m q2d.datasets.download_utils
```

Clones into:

- `./datasets/diagrams-repos/<owner>/<repo>`

### 3) Extract files

```bash
python3.12 -m q2d.datasets.extract_files
```

Writes:

- `./datasets/extracted_files.json`

### 4) Filter + deduplicate + sample

```bash
python3.12 -m q2d.datasets.filter
```

Writes:

- `./datasets/deduped.json`
- `./datasets/sampled.json`

### 5) (Optional) Generate developer queries

```bash
python3.12 q2d/generation/generate_questions_r1.py
# or
python3.12 q2d/generation/generate_questions_qwq.py
```

Writes:

- `./datasets/questions_r1.json` / `./datasets/questions_qwq.json`

> These scripts are meant to be edited for your own model endpoint + model name.

---

## Formats

### Graph JSON schema (authoritative)

See: `q2d/common/types.py`

The model outputs a `Graph`:

- `nodes`: classes / functions / variables / fields / methods
- `edges`: directed relationships
- `packages`: grouping / nesting

### Common dataset record shape

Most scripts operate on items like:

```json
{
  "language": "Python",
  "repo": "owner/name",
  "path": "path/in/repo.py",
  "code": "...",
  "query": "What does X do?",
  "version": "minimal",
  "diagram": { "...Graph JSON..." }
}
```

---

## Validate + render

### Validate graphs

- `q2d/checker.py` — structural validation + defect analysis for generated graphs

### Convert Graph JSON → diagram syntax

- `q2d/graph_to_plantuml.py` — converts graphs to:
  - PlantUML (`@startuml ... @enduml`)
  - Mermaid (class diagram + flowchart templates)

Rendering is external (PlantUML / Mermaid).

---

## Key entrypoints (what to run)

Inference:

- `q2d/generation/generate_diagrams_local_finetuned.py` — diagram generation via vLLM + LoRA (`finetuned_model`)
- `q2d/generation/generate_diagrams_local.py` — baseline generation via vLLM (no LoRA)
- `scripts/serve.sh` — starts vLLM server (OpenAI-compatible) with LoRA enabled

Training:

- `q2d/training/to_alpaca.py` — converts dataset JSON → Alpaca JSON for LLaMA-Factory
- `q2d/training/llama3_lora_sft.yaml` — LoRA SFT configuration
- `scripts/train.sh` — convenience wrapper (path may require fixing)

Data collection:

- `q2d/datasets/download_utils.py` — clones repos from `datasets/top_150.csv`
- `q2d/datasets/extract_files.py` — extracts source files into JSON
- `q2d/datasets/filter.py` — filters, dedupes, samples to target size

---

## Repo structure (annotated)

```text
.
├── Dockerfile
│   └── CUDA + Python + torch + vLLM + pip deps; installs package and sets CMD to scripts/start.sh
├── requirements.txt
│   └── Python deps for mining, generation, notebooks, training (LLaMA-Factory, etc.)
├── pyproject.toml
│   └── minimal build-system config (setuptools)
├── setup.py
│   └── installs package `q2d` + console script entrypoint `q2d` (see q2d/cli.py)
├── version.json
│   └── snapshot-style version metadata
├── build_lib.sh
│   └── dev helper: build wheel and reinstall locally (not required for Docker usage)

├── scripts/
│   ├── serve.sh
│   │   └── starts vLLM server with LoRA enabled (adapter: ./datasets/finetuned_model)
│   ├── start.sh
│   │   └── container entrypoint; optionally starts sshd if `SSHD=true` (then sleeps)
│   └── train.sh
│       └── convenience wrapper for llamafactory-cli (path may need adjustment)

├── datasets/
│   ├── test.json
│   │   └── small demo inputs for inference scripts
│   ├── claude_sonnet_synth.json
│   │   └── earlier synthetic dataset
│   ├── claude_sonnet_synth_fixed_final.json
│   │   └── corrected dataset used by training conversion
│   ├── sampled_ids.json
│   │   └── validation IDs (used by q2d/training/to_alpaca.py)
│   ├── lang_extensions.json
│   │   └── language → file extension mapping (used by extraction)
│   ├── annotations.csv
│   │   └── annotation metadata used during dataset work (see notebooks)
│   └── training/
│       └── dataset_info.json
│           └── LLaMA-Factory dataset registry for `diagrams_alpaca*` files
│
├── notebooks/
│   └── research notebooks for processing, manual fixing, and stats
│
├── images/
│   └── figures / visuals used for the paper/demo
│
└── q2d/
    ├── common/
    │   ├── types.py        # Pydantic schema: Graph / Node / Edge / Package (+ enums)
    │   ├── utils.py        # shared helpers (IDs, seeding, misc)
    │   └── cli_utils.py    # small helpers for script/CLI ergonomics
    ├── datasets/
    │   ├── download_utils.py       # clone repos listed in datasets/top_150.csv
    │   ├── extract_files.py        # traverse repos, extract source files
    │   ├── filter.py               # filter + deduplicate + sample
    │   ├── nearest_neighbours.py   # near-duplicate / similarity utilities
    │   ├── sample_val.py           # picks validation IDs → datasets/sampled_ids.json
    │   └── set_similarity_search/  # set-similarity index helpers used in dedup/NN
    ├── generation/
    │   ├── prompts.py                    # prompt templates used across scripts
    │   ├── generate_api.py               # shared “generate()” driver (batching, orchestration)
    │   ├── generate_utils.py             # OpenAI-compatible engine wrappers + parsing/retries
    │   ├── generate_diagrams_local.py    # baseline: JSON graphs from vLLM base model
    │   ├── generate_diagrams_local_finetuned.py  # LoRA: JSON graphs from vLLM + adapter
    │   ├── generate_diagrams_claude.py   # API-based diagram generation (if configured)
    │   ├── generate_questions_qwq.py     # optional query generation (requires endpoint config)
    │   └── generate_questions_r1.py      # optional query generation (requires endpoint config)
    ├── training/
    │   ├── to_alpaca.py            # dataset conversion → Alpaca JSON
    │   └── llama3_lora_sft.yaml    # LoRA SFT config (base model path, output_dir, etc.)
    ├── checker.py
    │   └── structural validation + defect analysis for generated graphs
    ├── graph_to_plantuml.py
    │   └── Graph JSON → PlantUML / Mermaid converters
    └── cli.py
        └── console entrypoint (minimal stub)
```

---

## Citation

```bibtex
@misc{baryshnikov2026query2diagram,
  title={Query2Diagram: Answering Developer Queries with UML Diagrams},
  author={Baryshnikov, Oleg and Alekseev, Anton M. and Nikolenko, Sergey I.},
  year={2026},
  eprint={2604.23816},
  archivePrefix={arXiv},
  primaryClass={cs.SE}
}
```
