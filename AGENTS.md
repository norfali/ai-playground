# Agent guidelines

## Project

SpeechLab is a small, open-source experiment for exploring local speech recognition and evaluation on Apple Silicon. It is a personal learning project, not a hosted product or production service.

Keep the project simple, focused and easy to understand. Prefer straightforward solutions over additional abstractions, dependencies or infrastructure.

Use **SpeechLab** for the application name and `speech-lab` for the repository and CLI command.

## Architecture

- `transcription.py` — MLX Whisper integration, audio validation, transcription and timing.
- `evaluation.py` — text normalisation, JiWER metrics, word alignment and safe HTML rendering.
- `app.py` — Gradio interface, session state and UI callbacks.
- `cli.py` — local application startup, configuration and telemetry controls.
- `styles.css` — interface styling and theme-specific overrides.
- `tests/` — evaluation edge cases, mocked inference and application behaviour.

Keep transcription and evaluation logic independent of Gradio. UI callbacks should remain thin.

## Engineering principles

- Use Python 3.12 and `uv`.
- Write clear, typed Python with small, focused functions.
- Follow existing project conventions before introducing new patterns.
- Prefer the smallest change that solves the problem.
- Avoid speculative features, unnecessary abstractions and premature optimisation.
- Preserve the original Whisper output; reference edits must not modify it.
- Keep evaluation results deterministic and document changes to normalisation, alignment or metric semantics.
- Escape user-provided text before rendering HTML.
- Communicate differences using text labels as well as colour.
- Prefer accessible native components and preserve keyboard navigation and visible focus indicators.
- Do not claim accessibility compliance without appropriate testing.

## Local-first principles

- Keep speech recognition local to the user's Mac.
- Do not introduce cloud inference, public sharing or unnecessary telemetry.
- Preserve the separation between transcription, evaluation and the UI.
- Avoid unnecessary dependencies, infrastructure and complexity.
- Never commit recordings, transcripts, model weights, secrets or caches.
- Document model downloads, temporary audio storage and recognition limitations accurately.

## Development

Install dependencies:

```bash
uv sync
```

Run the application:

```bash
uv run speech-lab
```

Run checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Keep `uv.lock` updated when dependencies change.

Real inference requires Apple Silicon macOS and FFmpeg. Automated tests must mock inference and must not download model weights.

## Working with agents

- Inspect relevant code and tests before making changes.
- Make targeted edits rather than unrelated refactors.
- Add or update tests when behaviour changes.
- Run relevant checks and report any failures or checks that could not be run.
- Keep README and other documentation aligned with actual behaviour.
- Do not commit, push or publish changes unless explicitly requested.
- Summarise what changed and highlight any important limitations or trade-offs.
