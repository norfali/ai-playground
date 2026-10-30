# AI Playground

A place for me to explore and dig deeper into the world of AI, try new ideas and see what works (and what doesn't).

Nothing too serious — just curiosity, code and a bit of trial and error.

## Explorations

| Project | What's it about? |
| --- | --- |
| [SpeechLab](./speech/) | Exploring local speech recognition with Whisper, comparing models and measuring transcription accuracy. |

More to come. RAG, agents, evals and whatever else catches my interest!

## Getting started

You'll need Python 3.12 and [uv](https://docs.astral.sh/uv/). Individual projects may have additional requirements.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/norfali/ai-playground.git
cd ai-playground
make sync
```

The projects share a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with a single Python environment and lockfile, while keeping their own dependencies and source code.

A simple `Makefile` provides shortcuts for running projects and common development tasks:

```bash
make               # Show available commands
make sync          # Sync workspace dependencies
make lint          # Check Python linting and formatting
make format        # Format Python code
make speech        # Launch SpeechLab
make speech-test   # Run tests
```

Run these commands from the repository root. See each project's README for further details.

## Legal

These are personal projects for learning and exploration, provided as-is without warranties. They're not affiliated with or endorsed by my employer.

The original code is released under the [MIT License](LICENSE). Third-party tools, models and datasets remain subject to their respective licences and terms.
