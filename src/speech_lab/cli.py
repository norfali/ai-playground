"""Entry point for the local speech-lab application."""

import os
import tempfile
from pathlib import Path


def main() -> None:
    # Configure privacy settings before importing Gradio.
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault(
        "GRADIO_TEMP_DIR",
        str(Path(tempfile.gettempdir()) / "speech-lab"),
    )

    from speech_lab.app import build_app, build_theme

    app = build_app()
    app.queue(default_concurrency_limit=1, max_size=8)
    try:
        app.launch(
            theme=build_theme(),
            css_paths=Path(__file__).with_name("styles.css"),
            server_name="127.0.0.1",
            share=False,
            inbrowser=False,
        )
    except KeyboardInterrupt:
        # Gradio handles the first Ctrl+C; a second can interrupt its shutdown wait.
        pass
