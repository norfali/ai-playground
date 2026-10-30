
.DEFAULT_GOAL := help

.PHONY: help sync lint format speech speech-test

help:
	@echo "AI Playground"
	@echo ""
	@echo "  make sync         Sync workspace dependencies"
	@echo "  make lint         Check Python linting and formatting"
	@echo "  make format       Format Python code"
	@echo "  make speech       Launch SpeechLab"
	@echo "  make speech-test  Run SpeechLab tests"

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
