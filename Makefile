
.DEFAULT_GOAL := help

.PHONY: help sync lint format speech speech-test rag rag-test rag-smoke rag-eval

help:
	@echo "AI Playground"
	@echo ""
	@echo "  make sync         Sync workspace dependencies"
	@echo "  make lint         Check Python linting and formatting"
	@echo "  make format       Format Python code"
	@echo "  make speech       Launch SpeechLab"
	@echo "  make speech-test  Run SpeechLab tests"
	@echo "  make rag          Launch RagLab"
	@echo "  make rag-test     Run RagLab tests"
	@echo "  make rag-smoke MODEL_DIR=/path/to/model [OUTPUT=rag/outputs/smoke-baseline-01.json]"
	@echo "                    Run the offline real-model smoke test"
	@echo "  make rag-eval [TOP_K=5] [OUTPUT=rag/outputs/retrieval-baseline-01.json]"
	@echo "                    Evaluate retrieval with cached embeddings (no LLM)"

sync:
	uv sync --all-packages

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .


# Speech Lab
# ========================= 
speech:
	uv run --package speech-lab speech-lab

speech-test:
	uv run --package speech-lab pytest speech/tests

# RAG Lab
# ========================= 

rag:
	uv run --package rag-lab rag-lab

rag-test:
	# Also filter SWIG's shutdown warning, emitted after pytest restores its filters.
	uv run --package rag-lab python \
		-W 'ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning' \
		-m pytest rag/tests

rag-smoke: OUTPUT ?= rag/outputs/smoke-baseline-01.json
rag-smoke:
	@test -n "$(MODEL_DIR)" || { echo 'Usage: make rag-smoke MODEL_DIR=/path/to/model [OUTPUT=/path/to/report.json]' >&2; exit 2; }
	uv run --offline --no-sync --package rag-lab --extra llm python -m rag_lab.smoke \
		--model-dir "$(MODEL_DIR)" --output "$(OUTPUT)"

rag-eval: TOP_K ?= 5
rag-eval: OUTPUT ?= rag/outputs/retrieval-baseline-01.json
rag-eval:
	uv run --offline --no-sync --package rag-lab python -m rag_lab.evaluation \
		--top-k "$(TOP_K)" --output "$(OUTPUT)"
