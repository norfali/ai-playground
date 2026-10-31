"""Lazy, shared local model; document state lives elsewhere."""

import os
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from rag_lab.chunking import TokenDiagnostics

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model_lock = Lock()


@lru_cache(maxsize=1)
def _load_model() -> "SentenceTransformer":
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_MODEL, device="cpu", trust_remote_code=False)


def encode_texts(texts: list[str]) -> NDArray[np.float32]:
    """Batch encode with unit length vectors, downloading model files only if needed."""
    if not texts or any(not text for text in texts):
        raise ValueError("Embedding input must contain non-empty text.")
    try:
        # Also serialize first load so concurrent sessions cannot load duplicate models.
        with _model_lock:
            return np.asarray(
                _load_model().encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    batch_size=32,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not load or run the local embedding model. On first use, internet access "
            "is needed to download model files; later runs use the local cache."
        ) from exc


def diagnose_texts(texts: list[str]) -> list[TokenDiagnostics]:
    """Count unpadded tokens once, using the loaded model's tokenizer and limit.

    Special tokens count toward the window. No model settings are changed.
    The supported MiniLM encoder uses plain text without an embedding prompt.
    """
    if not texts:
        return []
    with _model_lock:
        model = _load_model()
        limit = model.max_seq_length
        if not isinstance(limit, int) or limit <= 0:
            raise RuntimeError("The embedding model has no usable input token limit.")
        encoded = model.tokenizer(
            [text.strip() for text in texts],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [
            TokenDiagnostics(len(ids), min(len(ids), limit), limit) for ids in encoded["input_ids"]
        ]
