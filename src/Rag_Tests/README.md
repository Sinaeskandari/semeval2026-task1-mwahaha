# RAG Corpus-Size Sweep

This directory holds the **outputs** (`outputs_<model>_rag_<size>.jsonl`) of the
Wikipedia-retrieval corpus-size ablation reported in the paper. Sizes swept:
`5k, 10k, 15k, 20k, 25k` documents, for each of Gemma, Llama and Qwen.

The 15 near-identical generation scripts that used to live here (one per
`size × model` cell) differed only by a single constant. They have been replaced
by the parametrized entry point `humor.generate_rag`, which takes the corpus
size as a flag:

```bash
# Reproduce, e.g., the 15k Gemma run:
python -m humor.generate_rag \
    --model gemma \
    --corpus-size 15000 \
    --output src/Rag_Tests/Generation/15k/gemma/outputs_gemma_rag_15k.jsonl
```

Sweep all sizes for one model:

```bash
for size in 5000 10000 15000 20000 25000; do
    python -m humor.generate_rag --model qwen --corpus-size "$size" \
        --output "src/Rag_Tests/Generation/$((size/1000))k/qwen/outputs_qwen_rag_${size}.jsonl"
done
```

The original per-cell scripts remain available in the project's git history.
