"""CLI: base (no-retrieval) joke generation for any competitor model.

Replaces the per-model Colab scripts ``gen_gemma.py`` / ``gen_qwen.py`` /
``llama_generator.ipynb`` with a single argparse-driven entry point.

Example:
    python -m humor.generate --model qwen \\
        --input data/Datasets/task-a-en.tsv \\
        --output data/generated/Base/generated_jokes_qwen.jsonl
"""

from __future__ import annotations

import argparse
import logging

from humor import config, io_utils, models, parsing, prompts

logger = logging.getLogger("humor.generate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for base generation."""
    paths = config.Paths()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, choices=sorted(config.MODEL_REGISTRY),
        help="Competitor model to generate with.",
    )
    parser.add_argument(
        "--input", type=str, default=str(paths.input_tsv()),
        help="Path to the task TSV file.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (defaults to data/generated/Base/generated_jokes_<model>.jsonl).",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Optional local checkpoint directory to load instead of downloading.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N items (debugging).")
    return parser.parse_args(argv)


def generate(
    spec: config.ModelSpec,
    model,
    tokenizer,
    rows: list[dict],
    output_file,
    gen_cfg: config.GenerationConfig,
    *,
    max_english_retries: int = 3,
    limit: int | None = None,
) -> None:
    """Generate one joke per input row and append results to ``output_file``.

    Args:
        spec: Registry entry for the competitor (controls answer parsing and
            the English-only guard).
        model, tokenizer: Loaded model and tokenizer.
        rows: Task rows from :func:`humor.io_utils.read_task_inputs`.
        output_file: Destination JSONL path (append mode, resumable).
        gen_cfg: Sampling configuration.
        max_english_retries: Regeneration attempts when Chinese is detected
            (only for models with ``enforce_english``).
        limit: Optional cap on the number of newly generated items.
    """
    processed = io_utils.load_already_processed(output_file)
    produced = 0

    for row in rows:
        current_id = row["id"]
        if current_id in processed:
            continue
        if limit is not None and produced >= limit:
            break

        input_type, headline, word1, word2 = parsing.normalise_row(row)
        prompt = prompts.build_base_prompt(
            input_type, headline=headline, word1=word1, word2=word2,
            enforce_english=spec.enforce_english,
        )
        input_content = headline if input_type == "headline" else f"{word1}, {word2}"

        # Regenerate on code-switching for English-only models; a single pass otherwise.
        attempts = max_english_retries if spec.enforce_english else 1
        joke = ""
        for attempt in range(attempts):
            raw = models.generate_text(
                model, tokenizer, prompt,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                repetition_penalty=gen_cfg.repetition_penalty,
            )
            joke = parsing.extract_joke(raw, spec.answer_split_token)
            if not (spec.enforce_english and parsing.contains_chinese(joke)):
                break
            logger.warning("ID %s: Chinese detected (attempt %d/%d), retrying.",
                           current_id, attempt + 1, attempts)
        else:
            joke = "ERROR: GENERATION FAILED (CHINESE DETECTED)"

        io_utils.append_jsonl(output_file, {
            "id": current_id,
            "type": input_type,
            "input_original": input_content,
            "generated_joke": joke,
        })
        processed.add(current_id)
        produced += 1
        logger.info("ID %s: %s", current_id, joke[:60])

    logger.info("Finished. Generated %d new jokes -> %s", produced, output_file)


def main(argv: list[str] | None = None) -> None:
    """Entry point: load the model and run base generation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    spec = config.get_model_spec(args.model)

    paths = config.Paths()
    output_file = args.output or str(
        paths.generated_dir("Base") / f"generated_jokes_{spec.key}.jsonl"
    )

    defaults = config.GenerationConfig()
    gen_cfg = config.GenerationConfig(
        max_new_tokens=args.max_new_tokens or defaults.max_new_tokens,
        temperature=args.temperature or defaults.temperature,
        top_p=args.top_p or defaults.top_p,
        repetition_penalty=args.repetition_penalty or defaults.repetition_penalty,
    )

    models.maybe_mount_drive()
    rows = io_utils.read_task_inputs(args.input)
    source = models.resolve_model_source(spec.hf_model_id, args.model_path)
    model, tokenizer = models.load_model(source, max_seq_length=gen_cfg.max_seq_length)

    generate(spec, model, tokenizer, rows, output_file, gen_cfg, limit=args.limit)


if __name__ == "__main__":
    main()
