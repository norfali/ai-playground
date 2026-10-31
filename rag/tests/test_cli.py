"""Launch settings are local-only and configured before app construction."""

import os

from rag_lab import app, cli


def test_local_launch(monkeypatch):
    settings = {}

    class FakeApp:
        def queue(self, **kwargs):
            settings["queue"] = kwargs

        def launch(self, **kwargs):
            settings.update(kwargs)
            assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
            assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr(app, "build_app", FakeApp)
    monkeypatch.setattr(app, "build_theme", lambda: "theme")
    cli.main()
    assert settings["server_name"] == "127.0.0.1"
    assert settings["share"] is False
    assert settings["css"] == app.STYLES
    assert settings["inbrowser"] is False
