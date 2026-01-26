# SemEval 2026 Task 1 - MWAHAHA: A Competition on Humor Generation

This project implements a **text-based humor generation and evaluation pipeline** using three Large Language Models (LLMs) in a tournament format. The task focuses on generating jokes in response to news headlines, utilizing advanced techniques including Retrieval-Augmented Generation (RAG) and Direct Preference Optimization (DPO).

## Setup

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Sinaeskandari/semeval2026-task1-mwahaha
cd semeval2026-task1-mwahaha
pip install -r requirements.txt
```
### Running on Google Colab

All scripts are designed to run on Google Colab. When executing notebooks:
- Google Colab automatically provides GPU access
- `google.colab.drive` is pre-installed for Google Drive integration
- Models and outputs are saved to `/content/drive/MyDrive/Humor_Project/`

## Project Structure

```
semeval2026-task1-mwahaha/
├── data/
│   ├── raw/                     # Input data (news headlines)
│   ├── generated/               # Generated jokes by each model
│   └── evaluated/               # Tournament evaluation results
├── src/
│   ├── data generation/         # Data processing utilities
│   ├── generation/              # Joke generation scripts (Gemma, Llama, Qwen)
│   ├── fine_tuning/             # DPO fine-tuning implementation
│   └── evaluation/              # Judge models for tournament evaluation
└── README.md
```

## Methodology

### Step 1: Base Model Humor Generation

Generate jokes using three models based on news headlines from `data/raw/task-a-en.tsv`:

| Model | Script | Output |
|-------|--------|--------|
| Gemma-2 (9B) | `src/generation/gen_gemma.py` | `generated_jokes_gemma.jsonl` |
| Llama-3.1 (8B) | `src/generation/llama_generator.ipynb` | `generated_jokes_llama.json` |
| Qwen-2.5 (7B) | `src/generation/gen_qwen.py` | `generated_jokes_qwen.jsonl` |

All models use **4-bit quantization** and **Flash Attention** for efficiency.

### Step 2: Base Model Tournament Evaluation

Conduct a round-robin tournament where each pair of base models is evaluated by the third model as judge:

| Round | Competitors | Judge | Output |
|-------|-------------|-------|--------|
| 1 | Gemma vs Llama | Qwen | `result_competition_qwen_gemma_llama.jsonl` |
| 2 | Llama vs Qwen | Gemma | `result_competition_gemma_llama_qwen.jsonl` |
| 3 | Gemma vs Qwen | Llama | `result_competition_llama_gemma_qwen.jsonl` |

Run evaluation scripts in `src/evaluation/`:
```bash
python qwen_judge.py          # or qwen_judge_unsloth.py
python gemma_judge.py         # or gemma_judge_unsloth.py
python llama_judge.ipynb      # or llama_judge_unsloth.py
```

**Result**: Llama wins the base tournament.

### Step 3a: RAG-Enhanced Generation

Enhance all three models with **Retrieval-Augmented Generation** for improved contextual generation:

```bash
src/generation/Rag/
```

| Model | Output |
|-------|--------|
| Gemma-2 | `generated/Rag/outputs_gemma_rag.jsonl` |
| Llama-3.1 | `generated/Rag/outputs_Rag_lama.jsonl` |
| Qwen-2.5 | `generated/Rag/outputs_qwen_rag.jsonl` |

### Step 3b: DPO-Optimized Llama Generation

Apply **Direct Preference Optimization (DPO)** to fine-tune Llama (the base tournament winner):

```bash
src/fine_tuning/dpo_finetuning.ipynb
```

Output: `generated/dpo_generated_jokes.json`

### Step 4: RAG Model Tournament Evaluation

Conduct tournament with RAG-enhanced outputs using the same round-robin structure:

| Round | Competitors | Judge | Output |
|-------|-------------|-------|--------|
| 1 | Gemma-RAG vs Llama-RAG | Qwen-RAG | `evaluated/rag/result_rag_competition_qwen_*` |
| 2 | Llama-RAG vs Qwen-RAG | Gemma-RAG | `evaluated/rag/result_rag_competition_gemma_*` |
| 3 | Gemma-RAG vs Qwen-RAG | Llama-RAG | `evaluated/rag/result_rag_competition_llama_*` |

Run evaluation scripts in `src/evaluation/unsloth/`:
```bash
python qwen_judge_unsloth.py
python gemma_judge_unsloth.py
python llama_judge_unsloth.py
```

Results saved to `data/evaluated/rag/`

## Models

| Model | Repository | Version | Quantization | Max Seq |
|-------|-----------|---------|---------------|---------|
| Gemma-2 | HuggingFace | 9B Instruct | 4-bit | 2048 |
| Llama-3.1 | Meta | 8B Instruct | 4-bit | 2048 |
| Qwen-2.5 | Alibaba | 7B Instruct | 4-bit | 2048 |

## Results

Example evaluation summary from base tournament:
```json
{
  "judge_model": "google/gemma-2-9b-it",
  "comparison": "Gemma vs Llama",
  "gemma_wins": 450,
  "llama_wins": 550,
  "ties": 200,
  "total_evaluated": 1200,
  "winner": "Llama"
}
```

---

**Last Updated**: January 2026  
**Status**: In Progress (Steps 1-4 Complete)
