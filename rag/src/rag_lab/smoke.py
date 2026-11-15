"""Manual real-inference baseline: python -m rag_lab.smoke --help."""

import argparse
import hashlib
import json
import os
import platform
import re
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from rag_lab.embeddings import DEFAULT_MODEL
from rag_lab.generation import SYSTEM_PROMPT, LocalGenerator
from rag_lab.pipeline import QueryPipeline, RetrievalPipeline

# Fictional, deliberately small evidence. Keep missing facts out of the collection.
DOCUMENTS = {
    "harbour.txt": (
        "The fictional Brindle Harbour survey uses the vessel Lark. "
        "The survey starts on 14 June 2031. Its coordinator is Mira Vale.\n"
    ),
    "samples.txt": (
        "For the fictional Brindle Harbour survey, water samples are collected "
        "at North Pier every Tuesday. Samples are stored in amber glass bottles.\n"
    ),
}
SCENARIOS = (
    (
        "fully_supported",
        "Which vessel does the Brindle Harbour survey use, and when does the survey start?",
        "State Lark and 14 June 2031, with references to the supplied evidence.",
    ),
    (
        "partially_supported",
        "Who coordinates the Brindle Harbour survey, and what is the survey's total budget?",
        "State Mira Vale with a source reference; explicitly say the budget is not provided.",
    ),
    (
        "unsupported",
        "What is the emergency satellite phone number for the Brindle Harbour survey?",
        "Acknowledge insufficient evidence; do not invent a phone number.",
    ),
)


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("rag-lab", "mlx-lm", "mlx", "sentence-transformers", "transformers", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def run_smoke(model_directory: str) -> dict:
    """Run normal ingestion/query APIs offline, leaving quality decisions to a person.

    Call from a fresh process: offline flags must precede model-library imports.
    Tests replace the model boundaries, never the retrieval/query algorithms.
    """
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    generator = LocalGenerator(model_directory)
    model_path = Path(generator.identifier)
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": _versions(),
        "model_directory": generator.identifier,
        "model_config_sha256": (
            hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
            if (model_path / "config.json").is_file()
            else None
        ),
        "weight_files": [
            {"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(model_path.glob("*.safetensors"))
        ],
        "fixture": {
            name: {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
            for name, text in DOCUMENTS.items()
        },
        "settings": {
            "embedding_model": DEFAULT_MODEL,
            "chunk_size": 800,
            "overlap": 0,
            "strategy": "Character-based",
            "top_k": 2,
            "temperature": 0.0,
            "max_answer_tokens": generator.max_answer_tokens,
            "system_prompt": SYSTEM_PROMPT,
        },
        "ingestion_seconds": None,
        "error": None,
        "scenarios": [],
    }
    retrieval = RetrievalPipeline()
    try:
        with TemporaryDirectory(prefix="rag-smoke-") as directory:
            paths = []
            for name, text in DOCUMENTS.items():
                path = Path(directory) / name
                path.write_text(text, encoding="utf-8")
                paths.append(path)
            started = perf_counter()
            try:
                report["chunk_count"] = retrieval.index_documents(paths, chunk_size=800, overlap=0)
            finally:
                report["ingestion_seconds"] = perf_counter() - started
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        report["error"] = f"Ingestion failed (cached embeddings required): {exc}"
        return report

    pipeline = QueryPipeline(retrieval, generator)
    for name, question, expectation in SCENARIOS:
        observation = {
            "name": name,
            "question": question,
            "expected_behavior_for_human_review": expectation,
            "automated": None,
            "error": None,
            "human_review": {
                "evidence_sufficient_for_expected_behavior": None,
                "claims_supported_by_assembled_context": None,
                "citations_support_associated_claims": None,
                "meets_expected_behavior": None,
                "notes": "Pending human review; citation-label validity is not grounding.",
            },
        }
        report["scenarios"].append(observation)
        try:
            result = pipeline.run(question, top_k=2)
            output_tokens = None
            token_error = None
            if result.answer is not None and result.diagnostics.durations["generation"] is not None:
                try:
                    output_tokens = len(
                        generator.tokenizer.encode(result.answer, add_special_tokens=False)
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    token_error = str(exc)
            observation["automated"] = {
                "answer": result.answer,
                "status": result.status,
                "message": result.message,
                "retrieved_sources": [asdict(source) for source in result.results],
                "context": asdict(result.context),
                "citation_label_validation": {
                    "scope": "Label membership only; no factual verification",
                    "labels_used": list(
                        dict.fromkeys(re.findall(r"\[(S\d+)\]", result.answer or ""))
                    ),
                    "valid_labels": result.citations,
                    "warnings": result.warnings,
                },
                "diagnostics": asdict(result.diagnostics),
                "output_tokens_reencoded_estimate": output_tokens,
                "output_token_estimate_error": token_error,
                "inference_completed": (
                    result.answer is not None
                    and result.diagnostics.durations["generation"] is not None
                ),
            }
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            observation["error"] = str(exc)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Existing local MLX instruction model")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite previous observations, including manually reviewed reports.
    with args.output.open("x", encoding="utf-8") as output:
        report = run_smoke(args.model_dir)
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    completed = len(report["scenarios"]) == len(SCENARIOS) and all(
        row["automated"] is not None and row["automated"]["inference_completed"]
        for row in report["scenarios"]
    )
    print(f"Report: {args.output}")
    print(
        "Inference completed; human review pending."
        if completed
        else "Inference incomplete; inspect errors."
    )
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
