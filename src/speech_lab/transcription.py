"""Local speech transcription using MLX Whisper."""

import shutil
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter

MODELS = {
    "Small · multilingual (recommended)": "mlx-community/whisper-small-mlx",
    "Tiny · multilingual (faster)": "mlx-community/whisper-tiny-mlx",
}
DEFAULT_MODEL = MODELS["Small · multilingual (recommended)"]


class TranscriptionError(Exception):
    """A transcription error suitable for displaying to the user."""


@dataclass(frozen=True)
class Transcription:
    text: str
    model: str
    seconds: float
    audio_seconds: float

    @property
    def rtf(self) -> float | None:
        """Total processing seconds per decoded audio second, when defined."""
        if self.audio_seconds <= 0 or not isfinite(self.audio_seconds):
            return None
        return self.seconds / self.audio_seconds


def transcribe(audio_path: str | None, model: str = DEFAULT_MODEL) -> Transcription:
    """Transcribe an audio file locally using MLX Whisper."""
    if not audio_path:
        raise TranscriptionError("Upload or record audio before transcribing.")

    path = Path(audio_path)
    try:
        valid_file = path.is_file() and path.stat().st_size > 0
    except OSError as exc:
        raise TranscriptionError("The audio file could not be read.") from exc
    if not valid_file:
        raise TranscriptionError("The audio file is missing or empty.")

    if model not in MODELS.values():
        raise TranscriptionError("Choose an available Whisper model.")

    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("FFmpeg is required. Install it with: brew install ffmpeg")

    start = perf_counter()

    try:
        import mlx_whisper
        from mlx_whisper.audio import SAMPLE_RATE, load_audio

        try:
            audio = load_audio(str(path))
        except Exception as exc:
            raise TranscriptionError(
                "The audio could not be decoded. Check that it is a readable audio recording."
            ) from exc

        audio_seconds = len(audio) / SAMPLE_RATE
        if audio_seconds <= 0:
            raise TranscriptionError("The recording contains no audio samples.")

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model,
            verbose=None,
        )
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(
            "Transcription failed. Check the audio file, model download and FFmpeg installation."
        ) from exc

    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str):
        raise TranscriptionError("Whisper returned an unexpected result.")

    return Transcription(
        text=text,
        model=model,
        seconds=perf_counter() - start,
        audio_seconds=audio_seconds,
    )
