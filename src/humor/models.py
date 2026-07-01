"""Model loading utilities built on Unsloth's :class:`FastLanguageModel`.

All competitor and judge models are 4-bit quantised instruction-tuned LLMs
loaded through Unsloth, which transparently enables Flash-Attention and fast
inference kernels.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def maybe_mount_drive() -> None:
    """Mount Google Drive when running on Colab; a no-op elsewhere.

    The original scripts called ``drive.mount`` unconditionally (sometimes
    multiple times), which fails outside Colab. This guards the import so the
    same code runs locally.
    """
    if not (os.environ.get("COLAB_GPU") or Path("/content").exists()):
        return
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return
    logger.info("Mounting Google Drive ...")
    drive.mount("/content/drive")


def load_model(
    model_source: str,
    *,
    max_seq_length: int = 2048,
    for_inference: bool = True,
) -> tuple[Any, Any]:
    """Load a 4-bit model and tokenizer via Unsloth.

    Args:
        model_source: A Hugging Face model id (e.g.
            ``"unsloth/Qwen2.5-7B-Instruct-bnb-4bit"``) or a local directory
            containing a previously saved checkpoint.
        max_seq_length: Maximum context length for the loaded model.
        for_inference: If ``True``, enable Unsloth's fast inference path.

    Returns:
        A ``(model, tokenizer)`` tuple ready for generation.
    """
    # Imported lazily so the package can be imported (for docs/tests) on a
    # machine without a GPU or the Unsloth stack installed.
    from unsloth import FastLanguageModel

    logger.info("Loading model: %s", model_source)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    if for_inference:
        FastLanguageModel.for_inference(model)

    # Guard against tokenizers without a pad token (avoids attention-mask warnings).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
) -> str:
    """Run a single chat-template completion and return the decoded text.

    Wraps the boilerplate shared by every generation loop: build a one-turn
    user message, apply the chat template, sample, and decode (including the
    prompt, which the callers strip via :func:`humor.parsing.extract_joke`).

    Args:
        model: A loaded Unsloth model.
        tokenizer: The matching tokenizer.
        prompt: User-turn prompt content.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling probability mass.
        repetition_penalty: Penalty applied to repeated tokens.

    Returns:
        The decoded output text (special tokens removed).
    """
    import torch

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def resolve_model_source(hf_model_id: str, local_dir: str | Path | None) -> str:
    """Prefer a local checkpoint when available, else the Hub id.

    Downloading large checkpoints on every Colab session is slow; the original
    RAG scripts cached models on Drive. This helper reproduces that behaviour
    without hard-coding the path.
    """
    if local_dir and Path(local_dir).exists():
        logger.info("Using cached local checkpoint at %s", local_dir)
        return str(local_dir)
    return hf_model_id
