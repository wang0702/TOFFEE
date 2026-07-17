# TOFFEE

This repository contains the TOFFEE synthesis system and the released trajectory corpus. TOFFEE builds analytical tasks from real heterogeneous data environments, searches for high-quality trajectories with execution feedback, and adapts each step's tool, model tier, history length, and reasoning effort to the current analysis state. The resulting trajectories serve as supervised fine-tuning (SFT) data and in-context learning (ICL) demonstrations for data agents.

## Released Dataset (TOFFEE-30K)

The released trajectory corpus is hosted on Hugging Face:

`https://huggingface.co/datasets/toffee2027/toffee-30k`

- `data/toffee30k_swift.json` — the TOFFEE-30K corpus in Swift-format conversation JSON.

Each sample keeps the full multi-turn trajectory: reasoning turns, tool-call messages, tool outputs, and the tool metadata used during synthesis. The file is directly consumable as SFT data by MS-SWIFT or any trainer that reads conversation-format JSON, and as a retrieval pool for ICL demonstrations.

To download the dataset into this repository:

```
hf download toffee2027/toffee-30k toffee30k_swift.json --repo-type dataset --local-dir data
```

## System Overview

TOFFEE has three modules:

- **Task Pool Construction** loads each environment's sources into one scratch database, discovers a dependency graph by executing probe queries, samples and executes a path at a drawn complexity level, phrases the question after the answer key exists, and admits each task through execution checks (Accessible, Solvable, Stable, Nontrivial).
- **Trajectory Explorer** runs a tree search over analysis states. Every candidate step executes against the real environment, branch values come from observed outcomes, and shared prefixes are reused across tasks through a cross-task prefix DAG.
- **Learned Cost Model (LCM)** is a factored contextual LinUCB bandit that scores each feasible step choice (tool, model tier, history length, reasoning effort) under a cost-aware reward and sets the expansion width from its confidence intervals, learning online from execution feedback.

## Code Structure

- `cli.py` — orchestration entry point for synthesis runs.
- `config.py` — model tiers, budgets, and hyper-parameters; `TOFFEE_*` environment variables switch individual components to their ablation arms.
- `utils.py` — logging, budget tracking, and catalog loading.
- `generation/` — environment ingestion (`ingest.py`), dependency-graph construction and path sampling (`depgraph.py`), task synthesis and admission (`bottomup.py`), and SFT export (`assembler.py`).
- `search/` — Trajectory Explorer (`mcts.py`), Learned Cost Model (`bandit.py`), execution-grounded evaluation and acceptance (`evaluator.py`), and the cross-task prefix cache (`memoization.py`).
- `core/` — analysis state (`state.py`), step operators (`operators.py`), and the sandboxed tool executor (`executor.py`).
- `client/` — OpenRouter client wrapper.
- `agent_runtime/` — standalone runtime for executing trajectories: tool execution, context management, and checkpointing.

## Setup

```
pip install -r requirements.txt
export OPENROUTER_API_KEY=<your key>
```

## Source Data for Synthesis

The repository ships a small synthetic demo environment (`data/demo/`: three
CSV tables over a sales scenario) and a catalog describing it
(`data/data_source_catalog.json`), so the synthesis pipeline runs end to end
out of the box.

The environments used to build TOFFEE-30K come from seven public benchmarks;
synthesis reads only their data files and schemas, never their questions or
answers. To synthesize at scale, download the corpora from their official
releases and add one catalog entry per environment:

- BIRD — https://bird-bench.github.io
- Spider — https://yale-lily.github.io/spider
- SpreadsheetBench — https://github.com/RUCKBReasoning/SpreadsheetBench
- DABstep — https://huggingface.co/datasets/adyen/DAB-step
- LiveSQLBench — https://livesqlbench.ai
- DAComp — https://github.com/ByteDance-Seed/DAComp
- FDABench — https://fdabench.github.io

Catalog entry format:

```
{"source_id": "<id>", "source_group": "<group>",
 "data_file": "<path to .sqlite/.csv/.xlsx>", "extra_sources": ["<siblings>"]}
```

## Run

Clone this repository as `toffee` and run from its parent directory. To
synthesize one trajectory on the demo environment:

```
python -m toffee --run-name smoke --total 1 --workers 1
```

Outputs are written to `runs/<run-name>/`.
