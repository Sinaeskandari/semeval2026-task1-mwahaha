<div align="center">

# 🎭 MWAHAHA — Tournament-Based Humor Generation

### SemEval 2026 · Task 1 · Humor Generation from Headlines and Word Pairs

*Can Large Language Models compete to be funny? We stage a round-robin comedy
tournament between three open LLMs — then sharpen the winner with retrieval and
preference optimization.*

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg">
  <img alt="Unsloth" src="https://img.shields.io/badge/Unsloth-4bit-6f42c1.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

</div>

---

## ✨ Overview

**MWAHAHA** is a fully reproducible pipeline for *generating and evaluating jokes*
in response to two prompt types from SemEval 2026 Task 1:

- **Headlines** — write a standalone joke that satirizes a news headline.
- **Word pairs** — write a joke that creatively connects two given words.

Because "funniness" has no ground-truth label, we replace a fixed reward with a
**round-robin LLM tournament**: three competitors each generate jokes, and every
pair is judged by the *third* model as an impartial referee. The tournament both
*ranks* the models and *manufactures preference data* that we recycle to
fine-tune the winner.

### Core contributions

1. **Peer-judged tournament evaluation** — a rotating panel of LLM judges removes
   the need for human labels and cancels each model's self-preference bias.
2. **Two enhancement tracks** on the tournament winner: **Retrieval-Augmented
   Generation** (Wikipedia context) and **Direct Preference Optimization**
   (trained *directly on the tournament verdicts*), plus **LoRA** instruction
   tuning on model-distilled synthetic data.
3. **A corpus-size ablation** for RAG (5k → 25k Wikipedia documents).
4. **A refactored, installable toolkit** (`humor`) that reproduces every base,
   RAG and judging run from a single command.

### The competitors

| Model        | Checkpoint (4-bit, Unsloth)                    | Role                     |
|--------------|------------------------------------------------|--------------------------|
| Gemma-2 9B   | `unsloth/gemma-2-9b-it-bnb-4bit`               | Competitor + Judge       |
| Llama-3.1 8B | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`  | Competitor + Judge + FT target |
| Qwen-2.5 7B  | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`         | Competitor + Judge       |

All models run in **4-bit quantization** with Flash-Attention via Unsloth, so the
whole pipeline fits on a single Colab T4.

---

## 📦 Installation

```bash
# 1. Clone
git clone https://github.com/Sinaeskandari/semeval2026-task1-mwahaha
cd semeval2026-task1-mwahaha

# 2. Create an isolated environment (Python ≥ 3.10)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies and the local package
pip install -r requirements.txt
pip install -e .                   # exposes the `humor` package + console scripts
```

> **GPU required.** The generation/judging code loads 4-bit LLMs through
> [Unsloth](https://github.com/unslothai/unsloth) and needs a CUDA GPU. The
> pipeline is designed for Google Colab; see [Running on Colab](#-running-on-colab).

---

## 🗂️ Repository Structure

```
semeval2026-task1-mwahaha/
├── data/
│   ├── Datasets/                 # task-a-en.tsv (inputs) + synthetic_data.jsonl
│   ├── generated/                # jokes per stage: Base / Rag / Lora / Dpo
│   └── evaluated/                # tournament verdicts: Base / rag / Final_eval
│
├── src/
│   ├── humor/                    # ⭐ installable, refactored pipeline package
│   │   ├── config.py             #    model registry, paths, hyper-parameters
│   │   ├── io_utils.py           #    resumable JSONL, format-agnostic loading
│   │   ├── models.py             #    Unsloth model loading + generation helper
│   │   ├── prompts.py            #    base / RAG / judge prompt builders
│   │   ├── parsing.py            #    output cleaning + judge-verdict parsing
│   │   ├── retrieval.py          #    dense Wikipedia retriever (RAG)
│   │   ├── generate.py           #    CLI: base joke generation
│   │   ├── generate_rag.py       #    CLI: RAG joke generation (--corpus-size)
│   │   └── judge.py              #    CLI: paired-comparison tournament judging
│   │
│   ├── synthetic_data_generation/  # notebooks: sample + distill instruction data
│   ├── fine_tuning/
│   │   ├── Lora/                 # LoRA SFT experiments (EXP1/EXP2/NoReason/…)
│   │   └── Dpo/                  # DPO on tournament-mined preferences
│   ├── generation/               # notebooks: base + RAG + LoRA generation
│   ├── evaluation/               # llama_judge notebook (see humor.judge CLI)
│   └── Rag_Tests/                # RAG corpus-size ablation outputs (5k–25k)
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

The three CLIs below replace the former per-model Colab scripts. Each is
**resumable** — rerun after an interruption and completed items are skipped.

### 1. Base generation

```bash
# Generate jokes with any competitor (gemma / llama / qwen)
python -m humor.generate --model qwen \
    --input data/Datasets/task-a-en.tsv \
    --output data/generated/Base/generated_jokes_qwen.jsonl
```

### 2. Tournament judging

```bash
# Qwen referees Llama (A) vs Gemma (B)
python -m humor.judge --judge qwen \
    --competitor-a llama --joke-file-a data/generated/Base/generated_jokes_llama.json \
    --competitor-b gemma --joke-file-b data/generated/Base/generated_jokes_gemma.jsonl \
    --output  data/evaluated/Base/result_competition_qwen_llama_gemma.jsonl \
    --summary data/evaluated/Base/summary_qwen_llama_gemma.json
```

### 3. RAG generation (with corpus-size sweep)

```bash
# Single run
python -m humor.generate_rag --model gemma --corpus-size 25000 \
    --output data/generated/Rag/outputs_gemma_rag.jsonl

# Full ablation (5k → 25k)
for size in 5000 10000 15000 20000 25000; do
    python -m humor.generate_rag --model gemma --corpus-size "$size" \
        --output "data/generated/Rag/outputs_gemma_rag_${size}.jsonl"
done
```

### 4. Fine-tuning & synthetic data (notebooks)

Open in Colab or Jupyter, top-to-bottom:

| Stage                     | Notebook                                                        |
|---------------------------|-----------------------------------------------------------------|
| Sample ShortJokes corpus  | `src/synthetic_data_generation/sample_shortjokes.ipynb`         |
| Distill instruction data  | `src/synthetic_data_generation/generate_synthetic_data.ipynb`   |
| LoRA instruction tuning   | `src/fine_tuning/Lora/Llama_finetune_EXP1.ipynb`                 |
| DPO on tournament verdicts| `src/fine_tuning/Dpo/dpo_finetuning.ipynb`                       |

> `python -m humor.<tool> --help` lists every flag (sampling params, model paths,
> `--limit` for quick smoke tests).

---

## 🧪 Pipeline at a Glance

```
              ┌──────────────┐
 task-a-en →  │ Base Gen ×3  │ → jokes ──┐
              └──────────────┘           │
                                         ▼
                              ┌───────────────────────┐
                              │  Round-Robin Tournament│  (each model judges the
                              │   (peer LLM judges)    │   pair it did not write)
                              └───────────────────────┘
                                  │              │
                     winner: Llama│              │verdicts = preference pairs
                        ┌─────────┴───────┐      ▼
                        ▼                 ▼   ┌───────┐
                   ┌─────────┐      ┌──────┐ │  DPO  │→ Llama-DPO
                   │   RAG   │      │ LoRA │ └───────┘
                   └─────────┘      └──────┘
                        └───────┬───────┘
                                ▼
                        Final RAG Tournament → overall winner
```

---

## 📊 Main Results

> Numbers below are read directly from the committed evaluation summaries
> (`data/evaluated/**`). Regenerate them with `humor.judge`.

### Base tournament

| Comparison        | Judge | Winner    |
|-------------------|-------|-----------|
| Gemma vs Llama    | Qwen  | **Llama** |
| Llama vs Qwen     | Gemma | *(see `data/evaluated/Base/`)* |
| Gemma vs Qwen     | Llama | *(see `data/evaluated/Base/`)* |

➡️ **Llama-3.1-8B wins the base tournament** and becomes the fine-tuning target.

### Final RAG tournament (example verdict)

```json
{
  "judge_model": "google/gemma-2-9b-it",
  "comparison": "Llama_RAG vs Qwen_RAG",
  "llama_wins": 419,
  "qwen_wins": 779,
  "ties": 2,
  "total_evaluated": 1200,
  "overall_winner": "Qwen"
}
```

<!-- Add headline result tables / win-rate figures here as they are finalised. -->

📄 Full methodology, ablations and analysis: **[`SemEval_2026_Task_1_MWAHAHA.pdf`](SemEval_2026_Task_1_MWAHAHA.pdf)**.

---

## ☁️ Running on Colab

The notebooks were authored on Google Colab and mount Google Drive for
persistence. The `humor` package is Colab-aware:

- `humor.models.maybe_mount_drive()` mounts Drive only when running on Colab.
- Paths default to `/content/drive/MyDrive/Humor_Project` on Colab, or the
  repo-local `data/` directory otherwise. Override with `HUMOR_DATA_DIR` or the
  `--input` / `--output` flags.

---