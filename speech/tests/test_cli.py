"""Tests for local launch and interrupted shutdown."""

from unittest.mock import Mock

import pytest

from speech_lab import app, cli


def test_interrupted_shutdown_exits_without_traceback(monkeypatch, capsys):
    interface = Mock()
    interface.launch.side_effect = KeyboardInterrupt
    monkeypatch.setattr(app, "build_app", lambda: interface)
    monkeypatch.setattr(app, "build_theme", lambda: None)

    cli.main()

    interface.launch.assert_called_once()
    assert interface.launch.call_args.kwargs["server_name"] == "127.0.0.1"
    assert interface.launch.call_args.kwargs["share"] is False
    assert capsys.readouterr().err == ""


def test_launch_errors_are_not_silenced(monkeypatch):
    interface = Mock()
    interface.launch.side_effect = OSError("Port unavailable")
    monkeypatch.setattr(app, "build_app", lambda: interface)
    monkeypatch.setattr(app, "build_theme", lambda: None)

    with pytest.raises(OSError, match="Port unavailable"):
        cli.main()
