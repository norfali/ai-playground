"""Local Gradio interface for speech-lab."""

from html import escape
from math import isfinite

import gradio as gr
from gradio.themes import Base

from speech_lab.evaluation import (
    describe_metrics,
    evaluate,
    render_differences,
)
from speech_lab.transcription import (
    DEFAULT_MODEL,
    MODELS,
    Transcription,
    TranscriptionError,
    transcribe,
)


def build_theme() -> Base:
    """Use Gradio's mode-aware tokens and local system fonts; no font requests."""
    return Base(
        primary_hue="teal",
        secondary_hue="violet",
        neutral_hue="slate",
        font=["ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "monospace"],
        radius_size="md",
    ).set(
        body_background_fill="#fafaf7",
        body_background_fill_dark="#111b21",
        body_text_color="*neutral_800",
        body_text_color_dark="*neutral_100",
        body_text_color_subdued="*neutral_600",
        body_text_color_subdued_dark="*neutral_300",
        block_label_text_color="*neutral_600",
        block_label_text_color_dark="*neutral_300",
        input_placeholder_color="*neutral_500",
        input_placeholder_color_dark="*neutral_400",
        input_background_fill="white",
        input_background_fill_dark="*neutral_900",
        input_border_width="1px",
        input_border_color="*neutral_500",
        input_border_color_dark="*neutral_500",
        input_border_color_focus="*primary_700",
        input_border_color_focus_dark="*primary_300",
        link_text_color="*primary_700",
        link_text_color_dark="*primary_300",
        link_text_color_hover="*primary_800",
        link_text_color_hover_dark="*primary_200",
        button_primary_background_fill="*primary_700",
        button_primary_background_fill_dark="*primary_300",
        button_primary_background_fill_hover="*primary_800",
        button_primary_background_fill_hover_dark="*primary_200",
        button_primary_text_color="white",
        button_primary_text_color_dark="*neutral_950",
        button_primary_text_color_hover_dark="*neutral_950",
        button_primary_border_color="*primary_700",
        button_primary_border_color_dark="*primary_300",
        button_medium_padding="0.65rem 1.15rem",
        button_medium_text_weight="600",
        block_shadow="none",
        block_shadow_dark="none",
    )


def run_transcription(
    audio: str | None, model: str
) -> tuple[str | None, str, str, str, str, str, str, gr.Column]:
    try:
        result = transcribe(audio, model)
    except TranscriptionError as exc:
        return None, "", "", "", f"Error: {escape(str(exc))}", "", "", gr.Column(visible=False)

    status = "Transcription complete" if result.text.strip() else "No speech detected"
    summary = render_performance(result)

    metrics, differences = run_comparison(result.text, result.text)
    return (
        result.text,
        result.text,
        result.text,
        summary,
        status,
        metrics,
        differences,
        gr.Column(visible=True),
    )


def render_performance(result: Transcription) -> str:
    """Describe the completed run independently of the current model selection."""
    model_name = "Whisper (unknown model)"
    for label, repo in MODELS.items():
        if repo == result.model:
            model_name = f"Whisper {label.split(' ·', 1)[0]}"
            break
    rtf = (
        f"{result.rtf:.3f}"
        if result.rtf is not None and result.rtf >= 0 and isfinite(result.rtf)
        else "Unavailable"
    )
    speed = (
        result.audio_seconds / result.seconds
        if result.rtf is not None and result.seconds > 0 and isfinite(result.seconds)
        else None
    )
    speed_label = (
        f"{speed:.1f}× real time"
        if speed is not None and speed > 0 and isfinite(speed)
        else "Unavailable"
    )
    return (
        '<section aria-label="Local processing performance">'
        "<h3>Local processing performance</h3>"
        '<dl class="run-summary metric-list" aria-label="Latest transcription">'
        f"<div><dt>Model</dt><dd>{escape(model_name)}</dd></div>"
        f"<div><dt>Audio duration</dt><dd>{result.audio_seconds:.2f} s</dd></div>"
        "<div><dt>Processing time</dt>"
        f"<dd>{result.seconds:.2f} s</dd></div>"
        f"<div><dt>Processing speed</dt><dd>{speed_label}</dd></div></dl>"
        '<dl class="run-summary technical-metrics" aria-label="Technical measurement">'
        f"<div><dt>Real-time factor (RTF)</dt><dd>{rtf}</dd></div></dl>"
        "<p>Processing time includes decoding, model loading and any first-run download.</p>"
        "</section>"
    )


def run_comparison(raw: str | None, reference: str) -> tuple[str, str]:
    if raw is None:
        return "", ""

    result = evaluate(reference, raw)
    note = (
        '<p class="agreement-note">The reference is unchanged: zero errors reflect agreement '
        "with Whisper’s output, not verified accuracy.</p>"
        if reference == raw
        else ""
    )
    metrics = (
        '<section aria-label="Reference comparison"><h3>Reference comparison</h3>'
        f"{describe_metrics(result)}{note}</section>"
    )
    return metrics, render_differences(result)


def reset_transcription() -> tuple[None, str, str, str, str, str, str, gr.Column]:
    return None, "", "", "", "Ready to transcribe", "", "", gr.Column(visible=False)


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="SpeechLab — A closer listen.",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        with gr.Column(elem_id="speech-lab"):
            gr.HTML(
                '<header class="project-header">'
                '<div class="wordmark"><span class="sound-mark" aria-hidden="true">'
                "<i></i><i></i><i></i><i></i><i></i></span>"
                '<h1>Speech<span class="title-accent">Lab</span></h1></div>'
                '<a href="https://github.com/norfali/ai-playground/speech">View on GitHub</a>'
                '</header><div class="intro"><p class="eyebrow">LOCAL SPEECH PLAYGROUND</p>'
                '<p class="tagline">A closer listen.</p>'
                "<p>Turn audio into text, then explore what Whisper heard.</p></div>",
                elem_classes="project-intro",
            )

            gr.Markdown("## Start with a sound", elem_classes="section-heading")
            with gr.Row(elem_classes="audio-workflow", equal_height=False):
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio — upload or record",
                    scale=2,
                    min_width=280,
                    elem_classes="audio-input",
                )
                with gr.Column(scale=1, min_width=240, elem_classes="transcribe-controls"):
                    model = gr.Dropdown(
                        choices=list(MODELS.items()),
                        value=DEFAULT_MODEL,
                        label="Whisper model",
                    )
                    transcribe_button = gr.Button("Transcribe audio", variant="primary", size="md")
                    status = gr.HTML(
                        value="Upload a clip or record something to begin.",
                        html_template='<p role="status" aria-live="polite" '
                        'aria-atomic="true">${value}</p>',
                        elem_classes="status",
                    )

            raw_state = gr.State(None)

            gr.Markdown("## Read, listen, refine", elem_classes="section-heading")
            with gr.Row(equal_height=True, elem_classes="transcript-panels"):
                raw = gr.Textbox(
                    label="Whisper output · read only",
                    lines=6,
                    max_lines=6,
                    interactive=False,
                    placeholder="What Whisper hears will appear here.",
                    min_width=280,
                    buttons=["copy"],
                    elem_classes=["transcript", "transcript-readonly"],
                )
                reference = gr.Textbox(
                    label="Reference · editable",
                    lines=6,
                    max_lines=6,
                    interactive=True,
                    min_width=280,
                    placeholder="After transcribing, edit this copy to match the recording.",
                    elem_classes=["transcript", "transcript-editable"],
                )

            with gr.Column(visible=False, elem_classes="eval-section") as eval_section:
                gr.Markdown("## Evals: How did Whisper do?", elem_classes="section-heading")
                gr.Markdown("### Where they differed", elem_classes="eval-subheading")
                differences = gr.HTML(elem_classes="differences")
                with gr.Column(elem_classes="eval-results"):
                    metrics = gr.HTML(elem_classes="metrics", min_width=0)
                    summary = gr.HTML(elem_classes="run-details")

            gr.HTML(
                '<footer class="project-footer"><p>A little lab for listening.</p>'
                '<nav aria-label="Project links">'
                '<a href="https://www.fenka.co.uk">Fenka</a></nav></footer>',
                elem_classes="project-footer-container",
            )

            outputs = [
                raw_state,
                raw,
                reference,
                summary,
                status,
                metrics,
                differences,
                eval_section,
            ]

            transcribe_button.click(
                lambda: "Loading model / transcribing…",
                outputs=status,
                queue=False,
                show_progress="hidden",
            ).then(
                run_transcription,
                inputs=[audio, model],
                outputs=outputs,
                concurrency_limit=1,
                concurrency_id="speech",
                show_progress="minimal",
            )

            audio.change(
                reset_transcription,
                outputs=outputs,
                concurrency_limit=1,
                concurrency_id="speech",
            )

            reference.input(
                run_comparison,
                inputs=[raw_state, reference],
                outputs=[metrics, differences],
                trigger_mode="always_last",
                show_progress="hidden",
            )

    return app
