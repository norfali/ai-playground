"""Offline retrieval baseline: python -m rag_lab.evaluation --help (no answer generation)."""

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from rag_lab.chunking import STRATEGIES, validate_parameters
from rag_lab.documents import load_document
from rag_lab.embeddings import DEFAULT_MODEL
from rag_lab.pipeline import RetrievalPipeline

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "dataset.json"
DEFAULT_OUTPUT = Path("rag/outputs/retrieval-baseline-01.json")


@dataclass(frozen=True)
class EvaluationDataset:
    manifest: dict
    paths: tuple[Path, ...]
    source_ids: dict[str, str]
    fingerprint: str
    documents: tuple[dict, ...]


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def load_dataset(path: Path) -> EvaluationDataset:
    """Validate stable source IDs and verbatim evidence before loading any model.

    Quotes document the annotation; scoring uses the set of document IDs, not quotes.
    Unique basenames are required because the existing loader retains filenames.
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not _nonblank(manifest.get("id")):
        raise ValueError("Dataset must be an object with a nonblank id.")
    if type(manifest.get("version")) is not int or manifest["version"] <= 0:
        raise ValueError("Dataset version must be a positive integer.")
    for name in ("documents", "questions"):
        rows = manifest.get(name)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Dataset {name} must be a non-empty list.")
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or not _nonblank(row.get("id")) or row["id"] in ids:
                raise ValueError(f"Dataset {name} require unique nonblank ids.")
            ids.add(row["id"])
    paths, sources, documents, texts = [], {}, [], {}
    for row in manifest["documents"]:
        if not _nonblank(row.get("path")):
            raise ValueError("Each document needs a path.")
        source = (path.parent / row["path"]).resolve()
        if not source.is_relative_to(path.parent.resolve()) or source.suffix not in {".md", ".txt"}:
            raise ValueError("Evaluation sources must be TXT/Markdown files inside the dataset.")
        if source.name in sources:
            raise ValueError("Evaluation documents require unique filenames.")
        document = load_document(source)
        text = document.parts[0].text
        texts[row["id"]] = text
        sources[source.name] = row["id"]
        paths.append(source)
        documents.append({**row, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    for row in manifest["questions"]:
        if not _nonblank(row.get("question")) or type(row.get("unsupported")) is not bool:
            raise ValueError("Each question needs text and an explicit unsupported boolean.")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or bool(evidence) == row["unsupported"]:
            raise ValueError("Supported questions need evidence; unsupported questions need none.")
        if row["unsupported"] and not _nonblank(row.get("reason")):
            raise ValueError("Unsupported questions need a reason.")
        for item in evidence:
            if not isinstance(item, dict) or not _nonblank(item.get("document_id")):
                raise ValueError("Evidence needs a document_id.")
            if item["document_id"] not in texts or not _nonblank(item.get("quote")):
                raise ValueError("Evidence needs a known document_id and a nonblank quote.")
            if item["quote"] not in texts[item["document_id"]]:
                raise ValueError("Evidence quote must occur verbatim in the referenced document.")
    return EvaluationDataset(
        manifest,
        tuple(paths),
        sources,
        _fingerprint({"manifest": manifest, "documents": documents}),
        tuple(documents),
    )


def score_ranking(ranked_sources: list[str], relevant: set[str], k: int) -> dict | None:
    """Document relevance at raw chunk ranks; duplicate source hits earn no extra recall.

    Reciprocal rank is truncated at K (zero if the first hit is beyond K).
    Unsupported questions have no defined positive retrieval metrics.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("K must be a positive integer.")
    if not relevant:
        return None
    ranks = [rank for rank, source in enumerate(ranked_sources[:k], 1) if source in relevant]
    return {
        "hit_rate": float(bool(ranks)),
        "recall": len(set(ranked_sources[:k]) & relevant) / len(relevant),
        "reciprocal_rank": 1 / ranks[0] if ranks else 0.0,
    }


def aggregate_metrics(questions: list[dict], ks: list[int]) -> dict:
    """Macro average over successful supported questions; expose every denominator."""
    supported = [q for q in questions if not q["unsupported"]]
    completed = [q for q in supported if q["status"] == "completed"]
    by_k = {}
    for k in ks:
        scores = [q["metrics"][str(k)] for q in completed]
        by_k[str(k)] = {
            key: sum(score[field] for score in scores) / len(scores) if scores else None
            for key, field in (
                ("hit_rate", "hit_rate"),
                ("recall", "recall"),
                ("mrr", "reciprocal_rank"),
            )
        }
    return {
        "supported_count": len(supported),
        "scored_count": len(completed),
        "failed_or_skipped_supported_count": len(supported) - len(completed),
        "unsupported_count": len(questions) - len(supported),
        "by_k": by_k,
    }


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in (
        "rag-lab",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
        "pymupdf",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET,
    *,
    top_k: int = 5,
    chunk_size: int = 800,
    overlap: int = 120,
    strategy: str = "Character-based",
) -> dict:
    """Use normal ingestion/search offline in a fresh process with cached embeddings."""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("Top-k must be a positive integer.")
    validate_parameters(chunk_size, overlap)
    if strategy not in STRATEGIES:
        raise ValueError("Unknown chunking strategy.")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    started = perf_counter()
    ks = sorted({k for k in (1, 3, 5, top_k) if k <= top_k})
    settings = {
        "embedding_model": DEFAULT_MODEL,
        "embedding_device": "cpu",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "strategy": strategy,
        "top_k": top_k,
        "metric_ks": ks,
        "offline": True,
        "search": "exact cosine; stable ties in ingestion order",
    }
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": _versions(),
        "dataset": None,
        "settings": settings,
        "configuration_sha256": _fingerprint(settings),
        "scoring": {
            "relevance_unit": "document",
            "rank_unit": "chunk",
            "recall": "Unique relevant document IDs retrieved / expected relevant document IDs",
            "mrr": "Mean reciprocal first relevant chunk rank, truncated at each K",
            "scope": (
                "Document hits do not establish passage-level factual support. "
                "Quotes are annotation only."
            ),
            "unsupported": (
                "Excluded from positive metrics; no abstention or relevance threshold is evaluated."
            ),
        },
        "counts": {"documents": 0, "chunks": 0, "questions": 0},
        "timings_seconds": {"dataset_loading": None, "ingestion": None, "total": None},
        "error": None,
        "questions": [],
        "aggregate": None,
        "unsupported_questions": [],
    }
    try:
        before = perf_counter()
        try:
            dataset = load_dataset(dataset_path)
        finally:
            report["timings_seconds"]["dataset_loading"] = perf_counter() - before
        report["dataset"] = {
            "id": dataset.manifest["id"],
            "version": dataset.manifest["version"],
            "sha256": dataset.fingerprint,
            "documents": dataset.documents,
        }
        report["counts"].update(
            documents=len(dataset.paths), questions=len(dataset.manifest["questions"])
        )
        report["questions"] = [
            {
                **q,
                "status": "skipped",
                "error": None,
                "ranked_results": [],
                "metrics": None,
                "timings_seconds": {"query_embedding": None, "retrieval": None, "total": None},
            }
            for q in dataset.manifest["questions"]
        ]
        retrieval = RetrievalPipeline()
        before = perf_counter()
        try:
            report["counts"]["chunks"] = retrieval.index_documents(
                dataset.paths, chunk_size=chunk_size, overlap=overlap, strategy=strategy
            )
        finally:
            report["timings_seconds"]["ingestion"] = perf_counter() - before
        report["index_insights"] = asdict(retrieval.insights)
        for row in report["questions"]:
            query_started = perf_counter()
            relevant = {item["document_id"] for item in row["evidence"]}
            try:
                before = perf_counter()
                try:
                    vector = retrieval.embed_query(row["question"])
                finally:
                    row["timings_seconds"]["query_embedding"] = perf_counter() - before
                before = perf_counter()
                try:
                    results = retrieval.retrieve(vector, top_k)
                finally:
                    row["timings_seconds"]["retrieval"] = perf_counter() - before
                for rank, result in enumerate(results, 1):
                    source = dataset.source_ids[result.chunk.filename]
                    row["ranked_results"].append(
                        {
                            "rank": rank,
                            "document_id": source,
                            "chunk": asdict(result.chunk),
                            "cosine_similarity": result.score,
                            "document_relevant": source in relevant if relevant else None,
                        }
                    )
                ranked = [r["document_id"] for r in row["ranked_results"]]
                row["metrics"] = (
                    {str(k): score_ranking(ranked, relevant, k) for k in ks} if relevant else None
                )
                row["status"] = "completed"
            except (ValueError, OSError, RuntimeError, ImportError) as exc:
                row["status"], row["error"] = "error", str(exc)
            finally:
                row["timings_seconds"]["total"] = perf_counter() - query_started
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        report["error"] = str(exc)
        for row in report["questions"]:
            if row["status"] == "skipped":
                row["error"] = "Ingestion failed; query not executed."
    finally:
        report["aggregate"] = aggregate_metrics(report["questions"], ks)
        report["unsupported_questions"] = [q["id"] for q in report["questions"] if q["unsupported"]]
        report["timings_seconds"]["total"] = perf_counter() - started
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New JSON report path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--strategy", choices=STRATEGIES, default="Character-based")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        validate_parameters(args.chunk_size, args.overlap)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = args.output.open("x", encoding="utf-8")
    except FileExistsError:
        parser.exit(
            2,
            f"Report already exists: {args.output}\n"
            "Choose a new filename with OUTPUT=path/to/report.json (make rag-eval) "
            "or --output path/to/report.json (module CLI).\n",
        )
    with output:
        report = run_evaluation(
            args.dataset,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            strategy=args.strategy,
        )
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    print(f"Report: {args.output}")
    print(json.dumps(report["aggregate"], indent=2))
    completed = report["error"] is None and all(
        q["status"] == "completed" for q in report["questions"]
    )
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
