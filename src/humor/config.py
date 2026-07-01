"""Central configuration: model registry, generation defaults and paths.

Every hard-coded ``/content/drive/MyDrive/...`` path from the original Colab
scripts has been replaced by values that can be overridden on the command line
(see the ``generate``/``judge`` entry points) or through environment variables.

Environment variables
---------------------
``HUMOR_DATA_DIR``
    Base directory for inputs and outputs. Defaults to the repository's
    ``data`` directory, or ``/content/drive/MyDrive/Humor_Project`` when running
    on Google Colab (detected via the ``COLAB_GPU`` variable).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Static description of a competitor / judge model.

    Attributes:
        key: Short identifier used on the command line (e.g. ``"qwen"``).
        display_name: Human-readable name used in prompts and logs.
        hf_model_id: 4-bit Unsloth checkpoint on the Hugging Face Hub.
        answer_split_token: Chat-template marker after which the model's answer
            begins. Used as a fallback when the ``"Joke:"`` anchor is missing
            (``"model"`` for Gemma, ``"assistant"`` for Llama/Qwen).
        enforce_english: Whether to reject and regenerate outputs that contain
            Chinese characters (needed for Qwen, which occasionally code-switches).
    """

    key: str
    display_name: str
    hf_model_id: str
    answer_split_token: str
    enforce_english: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gemma": ModelSpec(
        key="gemma",
        display_name="Gemma-2-9B",
        hf_model_id="unsloth/gemma-2-9b-it-bnb-4bit",
        answer_split_token="model",
    ),
    "llama": ModelSpec(
        key="llama",
        display_name="Llama-3.1-8B",
        hf_model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        answer_split_token="assistant",
    ),
    "qwen": ModelSpec(
        key="qwen",
        display_name="Qwen-2.5-7B",
        hf_model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        answer_split_token="assistant",
        enforce_english=True,
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``key`` or raise a helpful error."""
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{key}'. Choose one of: {valid}.") from exc


# --------------------------------------------------------------------------- #
# Generation / retrieval hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters shared by base joke generation.

    Defaults reproduce the paper's winning configuration ("test 11").
    """

    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    max_seq_length: int = 2048


@dataclass(frozen=True)
class RagConfig:
    """Retrieval and sampling parameters for the RAG generation variant."""

    n_wiki_docs: int = 25_000
    retrieval_k: int = 4
    max_context_chars: int = 1_200
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    wiki_dataset: str = "not-lain/wikipedia"
    wiki_revision: str = "embedded"

    # Sampling (slightly tighter than base generation for constraint following).
    max_new_tokens: int = 64
    max_new_tokens_retry: int = 80
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_retries: int = 2


@dataclass(frozen=True)
class JudgeConfig:
    """Sampling parameters for the LLM judge (low temperature for consistency)."""

    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    max_seq_length: int = 2048


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _default_data_dir() -> Path:
    """Resolve the base data directory, honouring Colab and env overrides."""
    override = os.environ.get("HUMOR_DATA_DIR")
    if override:
        return Path(override)
    if os.environ.get("COLAB_GPU") or Path("/content/drive").exists():
        return Path("/content/drive/MyDrive/Humor_Project")
    # Repository-local default: <repo_root>/data
    return Path(__file__).resolve().parents[2] / "data"


@dataclass
class Paths:
    """Filesystem layout for inputs and outputs.

    A single ``data_dir`` is expanded into the concrete files used across the
    pipeline. Any field may be overridden individually if a non-standard layout
    is required.
    """

    data_dir: Path = field(default_factory=_default_data_dir)

    def input_tsv(self) -> Path:
        """Task input file (headlines and word pairs)."""
        return self.data_dir / "Datasets" / "task-a-en.tsv"

    def generated_dir(self, subdir: str = "Base") -> Path:
        """Directory holding generated jokes for a given experiment."""
        return self.data_dir / "generated" / subdir

    def evaluated_dir(self, subdir: str = "Base") -> Path:
        """Directory holding tournament evaluation results."""
        return self.data_dir / "evaluated" / subdir
