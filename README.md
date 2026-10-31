
# AI Playground

A place for me to explore and dig deeper into the world of AI, try new ideas and see what works (and what doesn't).

Nothing too serious — just curiosity, code and a bit of trial and error! 😀

[Fenka](https://www.fenka.co.uk)

## Explorations

### [SpeechLab](./speech/README.md)

**A closer listen.**

A local speech recognition playground for Apple Silicon. Transcribe audio with Whisper, compare the output against an editable reference and explore transcription accuracy, word-level differences and performance.

### [RagLab](./rag/README.md)

**A closer look at retrieval.**

A local document retrieval playground. Experiment with chunking strategies, embeddings and semantic search, then use Retrieval Insights to understand the results and explore how different settings affect retrieval.

RagLab currently focuses on retrieval rather than AI-generated answers.

More to explore: agents, evals and whatever else catches my interest!

## Getting started

You'll need Python 3.12 and [uv](https://docs.astral.sh/uv/). Individual projects may have additional requirements.

Clone the repository and install the workspace dependencies:

```bash
git clone https://github.com/norfali/ai-playground.git
cd ai-playground
make sync
```

Both projects share a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) and lockfile, while keeping their own dependencies and source code.

For VS Code with Pylance, open the repository root and select `.venv/bin/python`
using **Python: Select Interpreter**. The root `pyproject.toml` configures the
source paths for both projects so their imports resolve in the editor.

Run `make` to see the available commands for launching projects, running tests, linting and formatting.

For project-specific setup, usage and experiments, see the [SpeechLab](./speech/README.md) and [RagLab](./rag/README.md) READMEs.

## Legal

These are personal projects for learning and exploration, provided as-is without warranties. They're not affiliated with or endorsed by my employer.

The original code is released under the [MIT License](LICENSE). Third-party tools, models and datasets remain subject to their respective licences and terms.
