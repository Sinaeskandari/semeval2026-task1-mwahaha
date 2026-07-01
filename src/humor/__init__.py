"""MWAHAHA: Humor generation and tournament evaluation for SemEval 2026 Task 1.

This package provides a small, reusable toolkit for the paper's pipeline:

* :mod:`humor.config`    -- centralised model registry, paths and defaults.
* :mod:`humor.io_utils`  -- resumable JSONL writing and format-agnostic loading.
* :mod:`humor.models`    -- Unsloth model loading (Colab and local).
* :mod:`humor.prompts`   -- prompt builders for base and RAG generation.
* :mod:`humor.parsing`   -- output cleaning and judge-verdict parsing.
* :mod:`humor.retrieval` -- Wikipedia dense retriever used by the RAG variant.

The command-line entry points (``generate``, ``generate_rag`` and ``judge``)
supersede the per-model Colab scripts that previously lived under
``src/generation`` and ``src/evaluation``.
"""

__version__ = "1.0.0"

__all__ = [
    "config",
    "io_utils",
    "models",
    "prompts",
    "parsing",
    "retrieval",
]
