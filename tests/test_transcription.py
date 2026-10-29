"""Tests for local speech transcription."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from speech_lab import transcription
from speech_lab.transcription import DEFAULT_MODEL, Transcription, TranscriptionError, transcribe


@pytest.fixture
def backend(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"mock audio")

    infer = Mock(return_value={"text": "  Hello, world!\n"})

    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper",
        SimpleNamespace(transcribe=infer),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper.audio",
        SimpleNamespace(SAMPLE_RATE=16000, load_audio=Mock(return_value=range(40000))),
    )
    monkeypatch.setattr(transcription.shutil, "which", lambda _: "/mock/ffmpeg")

    return audio, infer


def test_transcription(backend, monkeypatch):
    audio, infer = backend
    times = iter([10.0, 11.25])
    monkeypatch.setattr(transcription, "perf_counter", lambda: next(times))

    result = transcribe(str(audio))

    assert result.text == "  Hello, world!\n"
    assert result.model == DEFAULT_MODEL
    assert result.seconds == 1.25
    assert result.audio_seconds == 2.5
    assert result.rtf == 0.5

    loader = sys.modules["mlx_whisper.audio"].load_audio
    loader.assert_called_once_with(str(audio))
    infer.assert_called_once_with(
        loader.return_value,
        path_or_hf_repo=DEFAULT_MODEL,
        verbose=None,
    )


@pytest.mark.parametrize("sample_count", [1, 16000, 123456])
def test_duration_uses_decoded_sample_count(backend, sample_count):
    audio, _ = backend
    sys.modules["mlx_whisper.audio"].load_audio.return_value = range(sample_count)

    assert transcribe(str(audio)).audio_seconds == sample_count / 16000


def test_zero_duration_audio(backend):
    audio, infer = backend
    sys.modules["mlx_whisper.audio"].load_audio.return_value = []

    with pytest.raises(TranscriptionError, match="no audio samples"):
        transcribe(str(audio))

    infer.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("malformed private file"), OSError("unreadable")])
def test_audio_decode_failure(backend, error):
    audio, infer = backend
    sys.modules["mlx_whisper.audio"].load_audio.side_effect = error

    with pytest.raises(TranscriptionError, match="could not be decoded") as caught:
        transcribe(str(audio))

    assert "private" not in str(caught.value)
    infer.assert_not_called()


def test_unreadable_file_metadata(backend, monkeypatch):
    audio, infer = backend

    def fail_stat(*_):
        raise PermissionError("private path")

    monkeypatch.setattr(transcription.Path, "stat", fail_stat)
    with pytest.raises(TranscriptionError, match="could not be read"):
        transcribe(str(audio))

    infer.assert_not_called()


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_rtf_unavailable_for_invalid_duration(duration):
    assert Transcription("", DEFAULT_MODEL, 1.0, duration).rtf is None


@pytest.mark.parametrize(("seconds", "rtf"), [(0, 0), (1, 0.5), (4, 2)])
def test_rtf(seconds, rtf):
    assert Transcription("", DEFAULT_MODEL, seconds, 2.0).rtf == rtf


@pytest.mark.parametrize("audio", [None, "", "/missing/speech-lab.wav"])
def test_missing_audio(audio):
    with pytest.raises(TranscriptionError):
        transcribe(audio)


def test_empty_audio(tmp_path):
    audio = tmp_path / "empty.wav"
    audio.touch()

    with pytest.raises(TranscriptionError, match="empty"):
        transcribe(str(audio))


def test_invalid_model(backend):
    audio, infer = backend

    with pytest.raises(TranscriptionError, match="available"):
        transcribe(str(audio), "unknown/model")

    infer.assert_not_called()


def test_missing_ffmpeg(backend, monkeypatch):
    audio, infer = backend

    monkeypatch.setattr(transcription.shutil, "which", lambda _: None)

    with pytest.raises(TranscriptionError, match="FFmpeg"):
        transcribe(str(audio))

    infer.assert_not_called()


def test_backend_failure(backend):
    audio, infer = backend
    infer.side_effect = RuntimeError("private path or transcript")

    with pytest.raises(TranscriptionError, match="Transcription failed") as error:
        transcribe(str(audio))

    assert "private" not in str(error.value)


@pytest.mark.parametrize("output", [{}, {"text": None}, "unexpected"])
def test_invalid_backend_response(backend, output):
    audio, infer = backend
    infer.return_value = output

    with pytest.raises(TranscriptionError, match="unexpected result"):
        transcribe(str(audio))
