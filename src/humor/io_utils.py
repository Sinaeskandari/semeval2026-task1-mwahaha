"""Input/output helpers: resumable JSONL and format-agnostic loading.

The original scripts re-implemented these routines in every file. They are
collected here so the CLIs stay thin and the behaviour is consistent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def read_task_inputs(input_path: str | Path) -> list[dict[str, Any]]:
    """Read the tab-separated task file into a list of row dictionaries.

    Args:
        input_path: Path to ``task-a-en.tsv`` (columns: ``id``, ``headline``,
            ``word1``, ``word2``).

    Returns:
        One dictionary per row, preserving the original column values.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    frame = pd.read_csv(input_path, sep="\t")
    return frame.to_dict("records")


def load_already_processed(output_file: str | Path) -> set[str]:
    """Return the set of ``id`` values already present in a JSONL output file.

    Enables the resume-on-restart behaviour used throughout the pipeline: a run
    that is interrupted can be launched again and will skip completed items.

    Args:
        output_file: Path to an append-mode JSONL file (may not yet exist).

    Returns:
        Set of already-generated ``id`` strings (empty if the file is absent).
    """
    output_file = Path(output_file)
    processed: set[str] = set()
    if not output_file.exists():
        return processed
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed line while scanning %s", output_file)
    if processed:
        logger.info("Resuming: found %d previously generated items.", len(processed))
    return processed


def append_jsonl(output_file: str | Path, record: dict[str, Any]) -> None:
    """Append a single record to a JSONL file, creating parents if needed.

    Writing one line at a time (rather than buffering) makes long generation
    runs crash-safe: whatever was produced before an interruption is preserved.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jokes(file_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a joke file (JSON or JSONL) into an ``id -> record`` dictionary.

    Different models saved their outputs in different shapes:

    * **Llama** wrote a nested JSON object ``{"ids": {id: record, ...}}``.
    * Some models wrote a flat JSON dict ``{id: record, ...}``.
    * Most wrote JSONL with one ``{"id": ...}`` record per line.

    This loader detects and normalises all three so the judge can treat every
    competitor identically.

    Args:
        file_path: Path to a ``.json`` or ``.jsonl`` joke file.

    Returns:
        Mapping from item ``id`` to its record. Empty if the file is missing.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("Joke file not found: %s", file_path)
        return {}

    # 1. Try whole-file JSON (Llama-style nested or flat dict).
    with file_path.open("r", encoding="utf-8") as handle:
        try:
            full = json.load(handle)
        except json.JSONDecodeError:
            full = None

    if isinstance(full, dict):
        if isinstance(full.get("ids"), dict):
            return full["ids"]
        first_value = next(iter(full.values()), None)
        if isinstance(first_value, dict):
            return full

    # 2. Fall back to line-delimited JSON.
    dataset: dict[str, dict[str, Any]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("id"):
                dataset[item["id"]] = item
    logger.info("Loaded %d items from %s", len(dataset), file_path)
    return dataset


def common_ids(*datasets: dict[str, Any]) -> list[str]:
    """Return the sorted ids present in *all* provided datasets."""
    if not datasets:
        return []
    shared: set[str] = set(datasets[0])
    for data in datasets[1:]:
        shared &= set(data)
    return sorted(shared)


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a JSON tournament summary (pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def iter_records(datasets: Iterable[dict[str, Any]], ids: list[str]):
    """Yield aligned records from several datasets for each id in ``ids``."""
    for current_id in ids:
        yield current_id, tuple(data[current_id] for data in datasets)
