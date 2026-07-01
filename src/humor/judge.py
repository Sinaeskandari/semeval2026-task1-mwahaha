"""CLI: paired-comparison tournament judging.

A judge model compares two competitors' jokes item-by-item and produces a JSON
verdict per item plus an aggregate summary. Replaces the per-judge Colab
scripts under ``src/evaluation`` and ``src/evaluation/unsloth``.

Example (Qwen judges Llama vs Gemma):
    python -m humor.judge --judge qwen \\
        --competitor-a llama --joke-file-a data/generated/Base/generated_jokes_llama.json \\
        --competitor-b gemma --joke-file-b data/generated/Base/generated_jokes_gemma.jsonl \\
        --output data/evaluated/Base/result_competition_qwen_llama_gemma.jsonl
"""

from __future__ import annotations

import argparse
import logging

from humor import config, io_utils, models, parsing, prompts

logger = logging.getLogger("humor.judge")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the tournament judge."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge", required=True, choices=sorted(config.MODEL_REGISTRY))
    parser.add_argument("--competitor-a", required=True, help="Display name for Joke A's source.")
    parser.add_argument("--competitor-b", required=True, help="Display name for Joke B's source.")
    parser.add_argument("--joke-file-a", required=True, help="Joke file (JSON/JSONL) for competitor A.")
    parser.add_argument("--joke-file-b", required=True, help="Joke file (JSON/JSONL) for competitor B.")
    parser.add_argument("--output", required=True, help="Per-item verdict JSONL output path.")
    parser.add_argument("--summary", default=None, help="Optional aggregate summary JSON output path.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def run_tournament(
    spec: config.ModelSpec,
    model,
    tokenizer,
    data_a: dict,
    data_b: dict,
    name_a: str,
    name_b: str,
    output_file,
    judge_cfg: config.JudgeConfig,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    """Compare two competitors over their common items and record verdicts.

    Args:
        spec: Judge model registry entry.
        model, tokenizer: Loaded judge model and tokenizer.
        data_a, data_b: ``id -> record`` mappings for the two competitors.
        name_a, name_b: Human-readable competitor names.
        output_file: Per-item verdict JSONL path.
        judge_cfg: Judge sampling configuration.
        limit: Optional cap on the number of comparisons.

    Returns:
        Scoreboard counts keyed by ``name_a``, ``name_b``, ``"Tie"`` and ``"Error"``.
    """
    ids = io_utils.common_ids(data_a, data_b)
    if limit is not None:
        ids = ids[:limit]
    scores = {name_a: 0, name_b: 0, "Tie": 0, "Error": 0}
    logger.info("Judging %d items with %s (%s vs %s).", len(ids), spec.display_name, name_a, name_b)

    for current_id in ids:
        entry_a, entry_b = data_a[current_id], data_b[current_id]
        input_type = entry_a.get("type", "headline")
        context = entry_a.get("input_original", entry_a.get("input", ""))
        joke_a = entry_a.get("generated_joke", "")
        joke_b = entry_b.get("generated_joke", "")

        prompt = prompts.build_judge_prompt(
            input_type, context, joke_a, joke_b, name_a=name_a, name_b=name_b,
        )
        raw = models.generate_text(
            model, tokenizer, prompt,
            max_new_tokens=judge_cfg.max_new_tokens,
            temperature=judge_cfg.temperature,
            top_p=judge_cfg.top_p,
        )
        # Drop any echoed instructions before parsing the verdict.
        if "OUTPUT FORMAT" in raw:
            raw = raw.split("OUTPUT FORMAT")[-1]

        verdict = parsing.parse_judge_verdict(raw)
        winner_code = verdict["winner"]
        if winner_code == "A":
            winner = name_a
        elif winner_code == "B":
            winner = name_b
        elif winner_code == "Tie":
            winner = "Tie"
        else:
            winner = "Error"
        scores[winner] += 1

        io_utils.append_jsonl(output_file, {
            "id": current_id,
            "content": context,
            "winner": winner,
            "reason": verdict["reasoning"],
            f"joke_{name_a.lower()}": joke_a,
            f"joke_{name_b.lower()}": joke_b,
        })
        logger.info(
            "%-10s | %-10s | %s:%d %s:%d Tie:%d",
            current_id, winner, name_a, scores[name_a], name_b, scores[name_b], scores["Tie"],
        )
    return scores


def main(argv: list[str] | None = None) -> None:
    """Entry point: load the judge, run the comparison and write the summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    spec = config.get_model_spec(args.judge)
    judge_cfg = config.JudgeConfig()

    data_a = io_utils.load_jokes(args.joke_file_a)
    data_b = io_utils.load_jokes(args.joke_file_b)
    if not data_a or not data_b:
        raise SystemExit("One or both joke files could not be loaded; aborting.")

    models.maybe_mount_drive()
    source = models.resolve_model_source(spec.hf_model_id, args.model_path)
    model, tokenizer = models.load_model(source, max_seq_length=judge_cfg.max_seq_length)

    scores = run_tournament(
        spec, model, tokenizer, data_a, data_b,
        args.competitor_a, args.competitor_b, args.output, judge_cfg, limit=args.limit,
    )

    total = sum(scores.values())
    winner = max((k for k in (args.competitor_a, args.competitor_b)), key=scores.get)
    summary = {
        "judge_model": spec.hf_model_id,
        "comparison": f"{args.competitor_a} vs {args.competitor_b}",
        "scores": scores,
        "total_evaluated": total,
        "winner": winner,
    }
    logger.info("Summary: %s", summary)
    if args.summary:
        io_utils.write_summary(args.summary, summary)


if __name__ == "__main__":
    main()
