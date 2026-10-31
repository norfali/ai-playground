
.DEFAULT_GOAL := help

.PHONY: help sync lint format speech speech-test rag rag-test

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
