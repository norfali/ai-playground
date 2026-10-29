"""Tests for the Gradio application callbacks."""

import pytest
from gradio import Column

from speech_lab import app
from speech_lab.transcription import Transcription, TranscriptionError


@pytest.mark.parametrize(
    ("used_model", "friendly_name"),
    list(zip(app.MODELS.values(), ("Whisper Small", "Whisper Tiny"), strict=True)),
)
@pytest.mark.parametrize(
    ("raw_text", "expected_wer"),
    [(" raw\n", "100.0%"), (" \tStraße\u00a0İSTANBUL\n", "200.0%")],
)
def test_transcription_preserves_raw_output(
    monkeypatch, used_model, friendly_name, raw_text, expected_wer
):
    monkeypatch.setattr(
        app,
        "transcribe",
        lambda *_: Transcription(raw_text, used_model, 1.25, 6.375),
    )

    selected_model = next(model for model in app.MODELS.values() if model != used_model)
    state, raw, reference, summary, status, metrics, differences, section = app.run_transcription(
        "audio.wav", selected_model
    )

    assert section.visible is True
    assert state == raw == reference == raw_text
    assert friendly_name in summary
    assert used_model not in summary
    assert "<dt>Model</dt>" in summary
    assert "<dt>Processing time</dt>" in summary
    assert "1.25 s" in summary
    assert "<dt>Audio duration</dt><dd>6.38 s</dd>" in summary
    assert "<dt>Processing speed</dt><dd>5.1× real time</dd>" in summary
    assert '<dl class="run-summary metric-list"' in summary
    assert '<dl class="run-summary technical-metrics"' in summary
    assert "<dt>Real-time factor (RTF)</dt><dd>0.196</dd>" in summary
    assert '<section aria-label="Local processing performance">' in summary
    assert "Processing time includes decoding, model loading and any first-run download." in summary
    assert '<section aria-label="Reference comparison">' in metrics
    assert "zero errors reflect agreement" in metrics
    assert "not verified accuracy" in metrics
    assert status == "Transcription complete"
    assert "<dt>WER</dt><dd>0.0%</dd>" in metrics
    assert "<dt>CER</dt><dd>0.0%</dd>" in metrics
    assert "<dt>Word errors</dt><dd>0</dd>" in metrics
    assert differences == "<p>No differences found</p>"

    def no_inference(*_):
        pytest.fail("Editing the reference must not rerun inference")

    monkeypatch.setattr(app, "transcribe", no_inference)
    metrics, differences = app.run_comparison(state, "corrected")

    assert f"<dt>WER</dt><dd>{expected_wer}</dd>" in metrics
    assert "agreement-note" not in metrics
    assert "Different word" in differences
    assert state == raw_text


def test_transcription_error_clears_results(monkeypatch):
    def fail(*_):
        raise TranscriptionError("Useful message")

    monkeypatch.setattr(app, "transcribe", fail)

    *outputs, section = app.run_transcription(None, "model")
    assert section.visible is False
    assert tuple(outputs) == (
        None,
        "",
        "",
        "",
        "Error: Useful message",
        "",
        "",
    )


def test_comparison_requires_transcription():
    metrics, differences = app.run_comparison(None, "")

    assert metrics == differences == ""


def test_transcription_error_escapes_status_html(monkeypatch):
    def fail(*_):
        raise TranscriptionError("Invalid <audio> & file")

    monkeypatch.setattr(app, "transcribe", fail)

    assert app.run_transcription(None, "model")[4] == "Error: Invalid &lt;audio&gt; &amp; file"


def test_empty_transcription_can_be_compared():
    metrics, differences = app.run_comparison("", "")

    assert "inserted 0 words" in metrics
    assert "no differences" in differences.lower()


def test_empty_reference_does_not_display_jiwer_counts_as_percentages():
    metrics, differences = app.run_comparison("two words", "")

    assert "<dt>WER</dt><dd>—</dd>" in metrics
    assert "<dt>CER</dt><dd>—</dd>" in metrics
    assert "<dt>Word errors</dt><dd>2</dd>" in metrics
    assert "Extra words: 2" in metrics
    assert "Extra word" in differences


def test_metrics_show_rates_and_all_word_error_counts():
    metrics, _ = app.run_comparison("one extra", "one missing word")

    assert '<dl class="metric-list"' in metrics
    assert "<dt>WER</dt><dd>66.7%</dd>" in metrics
    assert "<dt>Word errors</dt><dd>2</dd>" in metrics
    assert "Different words: 1 · Missing words: 1 · Extra words: 0" in metrics


def test_empty_transcription_status(monkeypatch):
    monkeypatch.setattr(app, "transcribe", lambda *_: Transcription(" \n", "model", 0.5, 2.0))

    state, raw, reference, _, status, _, _, section = app.run_transcription("audio.wav", "model")

    assert state == raw == reference == " \n"
    assert status == "No speech detected"
    assert section.visible is True


def test_undefined_rtf_is_labelled():
    summary = app.render_performance(Transcription("", "model", 0.5, 0))

    assert "<dt>Real-time factor (RTF)</dt><dd>Unavailable</dd>" in summary


@pytest.mark.parametrize(
    ("seconds", "duration"),
    [
        (0, 2),
        (-1, 2),
        (1, 0),
        (1, -1),
        (float("nan"), 2),
        (float("inf"), 2),
        (1, float("nan")),
        (1, float("inf")),
    ],
)
def test_processing_speed_unavailable(seconds, duration):
    summary = app.render_performance(Transcription("", "model", seconds, duration))

    assert "<dt>Processing speed</dt><dd>Unavailable</dd>" in summary


@pytest.mark.parametrize(("seconds", "speed"), [(1, "2.0"), (2, "1.0"), (4, "0.5")])
def test_processing_speed_relative_to_real_time(seconds, speed):
    summary = app.render_performance(Transcription("", "model", seconds, 2))

    assert f"<dt>Processing speed</dt><dd>{speed}× real time</dd>" in summary


@pytest.mark.parametrize("raw", ["", "Hello <world> & friends", " \n"])
def test_agreement_note_tracks_unchanged_reference(raw):
    metrics, _ = app.run_comparison(raw, raw)
    assert "zero errors reflect agreement" in metrics

    edited, _ = app.run_comparison(raw, raw + " edited")
    assert "agreement-note" not in edited

    restored, _ = app.run_comparison(raw, raw)
    assert "zero errors reflect agreement" in restored


def test_build_app():
    interface = app.build_app()

    assert interface.analytics_enabled is False
    assert interface.title == "SpeechLab — A closer listen."
    components = interface.config["components"]
    header = next(
        c["props"]["value"]
        for c in components
        if c["props"].get("elem_classes") == ["project-intro"]
    )
    assert '<a href="https://github.com/norfali/speech-lab">View on GitHub</a>' in header
    footer = next(
        c["props"]["value"]
        for c in components
        if c["props"].get("elem_classes") == ["project-footer-container"]
    )
    assert footer == (
        '<footer class="project-footer"><p>A little lab for listening.</p>'
        '<nav aria-label="Project links">'
        '<a href="https://www.fenka.co.uk">Fenka</a></nav></footer>'
    )
    headings = [c["props"].get("value") for c in components if c["type"] == "markdown"]
    assert all(
        f"## {heading}" in headings
        for heading in (
            "Start with a sound",
            "Read, listen, refine",
            "Evals: How did Whisper do?",
        )
    )
    assert "### Where they differed" in headings
    assert "Compare texts" not in [
        c["props"].get("value") for c in components if c["type"] == "button"
    ]
    assert not any(c["props"].get("elem_classes") == ["comparison-note"] for c in components)
    assert not any(c["type"] == "accordion" for c in components)
    section = next(c for c in components if c["props"].get("elem_classes") == ["eval-section"])
    assert section["props"]["visible"] is False
    section_block = interface.blocks[section["id"]]
    assert isinstance(section_block, Column)
    children = section_block.children
    assert children[0].get_config()["value"] == "## Evals: How did Whisper do?"
    assert children[1].get_config()["value"] == "### Where they differed"
    assert {tuple(child.elem_classes) for child in children} >= {
        ("eval-results",),
        ("differences",),
    }
    assert children[2].elem_classes == ["differences"]
    results_column = children[3]
    assert isinstance(results_column, Column)
    assert results_column.elem_classes == ["eval-results"]
    assert [child.elem_classes for child in results_column.children] == [
        ["metrics"],
        ["run-details"],
    ]
    transcripts = [c["props"] for c in components if c["type"] == "textbox"]
    assert all(t["lines"] == t["max_lines"] == 6 for t in transcripts)
    assert [t["interactive"] for t in transcripts] == [False, True]


def test_automatic_evaluation_event_wiring():
    interface = app.build_app()
    components = interface.config["components"]
    reference = next(
        c["id"] for c in components if c["props"].get("label") == "Reference · editable"
    )
    audio = next(c["id"] for c in components if c["type"] == "audio")
    metrics = next(c["id"] for c in components if c["props"].get("elem_classes") == ["metrics"])
    differences = next(
        c["id"] for c in components if c["props"].get("elem_classes") == ["differences"]
    )
    config = interface.config
    assert "dependencies" in config
    events = config["dependencies"]
    edit = next(event for event in events if (reference, "input") in event["targets"])
    assert interface.fns[edit["id"]].fn is app.run_comparison
    state = next(c["id"] for c in components if c["type"] == "state")
    model = next(c["id"] for c in components if c["type"] == "dropdown")
    summary = next(c["id"] for c in components if c["props"].get("elem_classes") == ["run-details"])
    assert edit["inputs"] == [state, reference]
    assert summary not in edit["outputs"]
    assert not any((model, "change") in event["targets"] for event in events)
    assert edit["outputs"] == [metrics, differences]
    assert edit["trigger_mode"] == "always_last"
    assert edit["show_progress"] == "hidden"
    assert not any((reference, "change") in event["targets"] for event in events)

    transcription = next(
        event for event in events if interface.fns[event["id"]].fn is app.run_transcription
    )
    reset = next(event for event in events if (audio, "change") in event["targets"])
    assert interface.fns[reset["id"]].fn is app.reset_transcription
    assert reset["outputs"] == transcription["outputs"]
    assert transcription["outputs"][3] == summary
    section = next(
        c["id"] for c in components if c["props"].get("elem_classes") == ["eval-section"]
    )
    assert transcription["outputs"][-3:] == [metrics, differences, section]
    assert section not in edit["outputs"]


def test_new_audio_clears_transcription_and_evaluation():
    *outputs, section = app.reset_transcription()
    assert tuple(outputs) == (None, "", "", "", "Ready to transcribe", "", "")
    assert section.visible is False
