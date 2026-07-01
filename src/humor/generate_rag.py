"""CLI: retrieval-augmented joke generation for any competitor model.

This single script replaces both the ``src/generation/Rag`` scripts and the
entire ``src/Rag_Tests/Generation/{5k,10k,15k,20k,25k}/{gemma,lamma,qwen}``
matrix. The corpus-size sweep is exposed as ``--corpus-size`` instead of being
duplicated across 15 near-identical files.

Example (reproduce the 15k Gemma run):
    python -m humor.generate_rag --model gemma --corpus-size 15000 \\
        --output data/generated/Rag/outputs_gemma_rag_15k.jsonl
"""

from __future__ import annotations

import argparse
import logging

from humor import config, io_utils, models, parsing, prompts
from humor.retrieval import WikipediaRetriever

logger = logging.getLogger("humor.generate_rag")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for RAG generation."""
    paths = config.Paths()
    rag_defaults = config.RagConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, choices=sorted(config.MODEL_REGISTRY),
        help="Competitor model to generate with.",
    )
    parser.add_argument("--input", type=str, default=str(paths.input_tsv()))
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (defaults to data/generated/Rag/outputs_<model>_rag.jsonl).",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--corpus-size", type=int, default=rag_defaults.n_wiki_docs,
        help="Number of Wikipedia docs to index (the paper sweeps 5000-25000).",
    )
    parser.add_argument("--retrieval-k", type=int, default=rag_defaults.retrieval_k)
    parser.add_argument("--max-context-chars", type=int, default=rag_defaults.max_context_chars)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def _retrieval_query(input_type: str, headline: str, word1: str, word2: str) -> str:
    """Build the retrieval query for a given input (headline vs word pair)."""
    if input_type == "headline":
        return f"Background facts and context about: {headline}"
    return f"Meaning, usage, and related concepts for: {word1} and {word2}"


def generate_rag(
    spec: config.ModelSpec,
    model,
    tokenizer,
    retriever: WikipediaRetriever,
    rows: list[dict],
    output_file,
    rag_cfg: config.RagConfig,
    *,
    limit: int | None = None,
) -> None:
    """Generate one RAG-conditioned joke per row and append to ``output_file``.

    For word-pair tasks the joke is regenerated (up to ``rag_cfg.max_retries``)
    until it contains both required words; the last attempt is kept regardless.

    Args:
        spec: Competitor registry entry.
        model, tokenizer: Loaded model and tokenizer.
        retriever: Initialised Wikipedia retriever.
        rows: Task rows.
        output_file: Destination JSONL path (resumable).
        rag_cfg: Retrieval and sampling configuration.
        limit: Optional cap on newly generated items.
    """
    import torch

    processed = io_utils.load_already_processed(output_file)
    produced = 0

    for index, row in enumerate(rows):
        current_id = row["id"]
        if current_id in processed:
            continue
        if limit is not None and produced >= limit:
            break

        input_type, headline, word1, word2 = parsing.normalise_row(row)
        input_content = headline if input_type == "headline" else f"{word1}, {word2}"

        query = _retrieval_query(input_type, headline, word1, word2)
        context = retriever.retrieve(query, k=rag_cfg.retrieval_k, max_chars=rag_cfg.max_context_chars)
        prompt = prompts.build_rag_prompt(
            input_type, context, headline=headline, word1=word1, word2=word2,
        )

        joke = ""
        for attempt in range(rag_cfg.max_retries):
            reminder = ""
            if attempt > 0 and input_type == "word-pair":
                reminder = f"\n\nREMINDER: You MUST include the words '{word1}' and '{word2}' in the joke."
            max_tokens = rag_cfg.max_new_tokens if attempt == 0 else rag_cfg.max_new_tokens_retry

            raw = models.generate_text(
                model, tokenizer, prompt + reminder,
                max_new_tokens=max_tokens,
                temperature=rag_cfg.temperature,
                top_p=rag_cfg.top_p,
                repetition_penalty=rag_cfg.repetition_penalty,
            )
            joke = parsing.extract_joke(raw, spec.answer_split_token)

            # Accept immediately for headlines; enforce word inclusion otherwise.
            if input_type != "word-pair":
                break
            if word1.lower() in joke.lower() and word2.lower() in joke.lower():
                break

        io_utils.append_jsonl(output_file, {
            "id": current_id,
            "type": input_type,
            "input_original": input_content,
            "retrieved_context": context,
            "generated_joke": joke or "Error: Generation failed.",
        })
        processed.add(current_id)
        produced += 1
        logger.info("ID %s: %s", current_id, joke[:60])

        # Periodically release cached GPU memory during long runs.
        if (index + 1) % 50 == 0:
            torch.cuda.empty_cache()

    logger.info("Finished. Generated %d new RAG jokes -> %s", produced, output_file)


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the retriever, load the model and run RAG generation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    spec = config.get_model_spec(args.model)

    paths = config.Paths()
    output_file = args.output or str(
        paths.generated_dir("Rag") / f"outputs_{spec.key}_rag.jsonl"
    )

    defaults = config.RagConfig()
    rag_cfg = config.RagConfig(
        n_wiki_docs=args.corpus_size,
        retrieval_k=args.retrieval_k,
        max_context_chars=args.max_context_chars,
    )

    models.maybe_mount_drive()
    rows = io_utils.read_task_inputs(args.input)
    retriever = WikipediaRetriever(
        n_docs=rag_cfg.n_wiki_docs,
        embedding_model=defaults.embedding_model,
        dataset=defaults.wiki_dataset,
        revision=defaults.wiki_revision,
    )
    source = models.resolve_model_source(spec.hf_model_id, args.model_path)
    model, tokenizer = models.load_model(source, max_seq_length=2048)

    generate_rag(spec, model, tokenizer, retriever, rows, output_file, rag_cfg, limit=args.limit)


if __name__ == "__main__":
    main()
