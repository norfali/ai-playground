# Agent guidelines

## Project

AI Playground is a small, open-source monorepo for exploring AI technologies, workflows and evaluation through practical, local-first experiments.

It is a personal learning project, not a hosted product or production service.

The repository currently contains:

- **SpeechLab** (`speech-lab/`) — local speech recognition and accuracy evaluation using MLX Whisper on Apple Silicon.
- **RagLab** (`rag/`) — document chunking, embeddings, semantic retrieval and retrieval insights, with scope for exploring RAG pipelines and evaluation.

Keep each lab focused, understandable and independently maintainable. Prefer straightforward solutions over additional abstractions, dependencies or infrastructure.

Follow the existing naming conventions: **SpeechLab** and **RagLab** for application names, with `speech-lab` and `rag` for their respective project directories.

## Architecture

- The root `pyproject.toml` and `uv.lock` manage the shared Python workspace and dependencies.
- The root `Makefile` provides common development commands.
- Each lab owns its application code, tests, documentation and project-specific dependencies.
- Shared functionality should remain within its respective lab unless there is a demonstrated need for reuse.

### SpeechLab

- Uses MLX Whisper for local speech recognition.
- Uses JiWER for transcription accuracy evaluation.
- Provides a Gradio interface for transcription, comparison and evaluation.
- Keeps transcription and evaluation logic independent of the UI.
- Requires Apple Silicon macOS and FFmpeg for real inference.

Preserve the original Whisper output; reference edits must not modify it. Keep evaluation results deterministic and document changes to normalisation, alignment or metric semantics.

### RagLab

- Supports local document processing, chunking and semantic retrieval.
- Uses sentence-transformers for embeddings.
- Provides a Gradio interface for exploring retrieved chunks and retrieval insights.
- Keeps document processing, embeddings and retrieval logic independent of the UI.
- Should support experimentation with RAG pipelines, evaluation and observability without unnecessary infrastructure.

Prefer explicit, independently testable pipeline stages. Introduce orchestration frameworks only when their benefits justify the additional complexity.

## Engineering principles

- Use Python 3.12+ within the repository's declared version constraints and `uv` for dependency management.
- Write clear, typed Python with small, focused functions.
- Follow existing project conventions before introducing new patterns.
- Prefer the smallest change that solves the problem.
- Avoid speculative features, unnecessary abstractions and premature optimisation.
- Keep application logic separate from Gradio callbacks.
- Make evaluation reproducible and document changes to metric semantics.
- Escape user-provided text before rendering HTML.
- Communicate differences using text labels as well as colour.
- Prefer accessible native components and preserve keyboard navigation and visible focus indicators.
- Do not claim accessibility compliance without appropriate testing.

## Local-first principles

- Keep inference, embeddings, retrieval and evaluation local.
- Do not introduce cloud inference, hosted APIs, public sharing or unnecessary telemetry.
- Avoid unnecessary infrastructure, databases, background services and deployment tooling.
- Do not introduce GitHub Actions or CI/CD unless explicitly requested.
- Never commit user documents, recordings, transcripts, model weights, secrets, generated indexes or caches.
- Document model requirements, downloads, temporary storage and known limitations accurately.

## Development

Run commands from the repository root.

Install dependencies:

```bash
make sync
```

Launch applications:

```bash
make speech
make rag
```

Run checks:

```bash
make lint
make format
make speech-test
make rag-test
```

Keep `uv.lock` updated when dependencies change.

Automated tests should use fixtures or mocks where appropriate and must not download model weights or require live inference.

## Working with agents

- Inspect the relevant project code, tests and documentation before making changes.
- Respect the monorepo structure and existing development conventions.
- Make targeted edits rather than unrelated refactors.
- Avoid modifying sibling projects unless the change genuinely requires it.
- Add or update tests when behaviour changes.
- Run relevant checks and report failures or checks that could not be run.
- Keep README files and other documentation aligned with actual behaviour.
- Explain significant architectural decisions and new dependencies.
- Do not commit, push or publish changes unless explicitly requested.
- Summarise what changed and highlight important limitations or trade-offs.