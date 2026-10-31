"""Launch RagLab locally with telemetry and public sharing disabled."""

import os
import tempfile
from pathlib import Path


def main() -> None:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(tempfile.gettempdir()) / "rag-lab"))

    from rag_lab.app import STYLES, build_app, build_theme

    app = build_app()
    app.queue(default_concurrency_limit=1, max_size=8)
    try:
        app.launch(
            theme=build_theme(),
            css=STYLES,
            server_name="127.0.0.1",
            share=False,
            inbrowser=False,
        )
    except KeyboardInterrupt:
        pass
