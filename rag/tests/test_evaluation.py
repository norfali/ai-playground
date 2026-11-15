"""Deterministic retrieval metrics and reports; never assert real-model rankings."""

import copy
import json
import os

import pytest

from rag_lab import evaluation
from rag_lab.pipeline import RetrievalPipeline


def test_metrics_with_multiple_sources_duplicates_and_cutoffs():
    ranking = ["distractor", "a", "a", "b", "other"]
    relevant = {"a", "b"}
    assert evaluation.score_ranking(ranking, relevant, 1) == {
        "hit_rate": 0,
        "recall": 0,
        "reciprocal_rank": 0,
    }
    assert evaluation.score_ranking(ranking, relevant, 3) == {
        "hit_rate": 1,
        "recall": 0.5,
        "reciprocal_rank": 0.5,
    }
    assert evaluation.score_ranking(ranking, relevant, 5) == {
        "hit_rate": 1,
        "recall": 1,
        "reciprocal_rank": 0.5,
    }
    assert evaluation.score_ranking(["x", "y", "a"], {"a"}, 2)["reciprocal_rank"] == 0
    assert evaluation.score_ranking([], {"a"}, 5)["recall"] == 0
    assert evaluation.score_ranking(["a"], {"a"}, 5)["recall"] == 1
    assert evaluation.score_ranking(["a"], set(), 3) is None


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_metric_cutoff(k):
    with pytest.raises(ValueError, match="positive"):
        evaluation.score_ranking([], {"a"}, k)


def test_macro_averaging_excludes_unsupported_errors_and_skips():
    rows = [
        {
            "unsupported": False,
            "status": "completed",
            "metrics": {"3": evaluation.score_ranking(["a"], {"a", "b"}, 3)},
        },
        {
            "unsupported": False,
            "status": "completed",
            "metrics": {"3": evaluation.score_ranking(["x", "b"], {"b"}, 3)},
        },
        {"unsupported": True, "status": "completed", "metrics": None},
        {"unsupported": False, "status": "error", "metrics": None},
        {"unsupported": False, "status": "skipped", "metrics": None},
    ]
    aggregate = evaluation.aggregate_metrics(rows, [3])
    assert aggregate["scored_count"] == 2
    assert aggregate["supported_count"] == 4
    assert aggregate["failed_or_skipped_supported_count"] == 2
    assert aggregate["unsupported_count"] == 1
    assert aggregate["by_k"]["3"] == {"hit_rate": 1, "recall": 0.75, "mrr": 0.75}
    assert evaluation.aggregate_metrics([], [1])["by_k"]["1"]["mrr"] is None


@pytest.fixture
def dataset_path(tmp_path):
    (tmp_path / "alpha.md").write_text("Alpha station. Blue boat. Safe harbour. " * 8)
    (tmp_path / "beta.md").write_text("Beta station. Red boat. Quiet harbour. " * 8)
    manifest = {
        "id": "test-data",
        "version": 1,
        "documents": [{"id": "alpha", "path": "alpha.md"}, {"id": "beta", "path": "beta.md"}],
        "questions": [
            {
                "id": "single",
                "question": "Where is the blue boat?",
                "unsupported": False,
                "evidence": [{"document_id": "alpha", "quote": "Blue boat."}],
            },
            {
                "id": "multiple",
                "question": "Which boats are at both stations?",
                "unsupported": False,
                "evidence": [
                    {"document_id": "alpha", "quote": "Blue boat."},
                    {"document_id": "beta", "quote": "Red boat."},
                ],
            },
            {
                "id": "missing",
                "question": "Who built the boats?",
                "unsupported": True,
                "evidence": [],
                "reason": "Builders are not recorded.",
            },
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def fake_embeddings(monkeypatch):
    def encode(texts):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        return [[1, 0] for _ in texts]

    monkeypatch.setattr(evaluation, "RetrievalPipeline", lambda: RetrievalPipeline(encode))


def test_report_and_cli(dataset_path, fake_embeddings, tmp_path, capsys, monkeypatch):
    output = tmp_path / "reports" / "baseline.json"
    args = ["--dataset", str(dataset_path), "--output", str(output)]
    assert evaluation.main(args) == 0
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["started_at_utc"] and report["versions"]["numpy"]
    assert report["error"] is None
    assert report["counts"] == {"documents": 2, "chunks": 2, "questions": 3}
    assert len(report["dataset"]["sha256"]) == 64
    assert len(report["configuration_sha256"]) == 64
    assert report["scoring"]["relevance_unit"] == "document"
    assert report["aggregate"]["scored_count"] == 2
    assert report["unsupported_questions"] == ["missing"]
    first = report["questions"][0]
    assert first["metrics"]["1"]["hit_rate"] == 1
    assert first["ranked_results"][0]["document_id"] == "alpha"
    assert first["ranked_results"][0]["document_relevant"] is True
    assert first["ranked_results"][0]["chunk"]["text"].startswith("Alpha station.")
    assert first["ranked_results"][1]["document_relevant"] is False
    assert first["ranked_results"][0]["cosine_similarity"] == pytest.approx(1)
    assert all(value >= 0 for value in first["timings_seconds"].values())
    assert all(value >= 0 for value in report["timings_seconds"].values())
    unsupported = report["questions"][2]
    assert unsupported["metrics"] is None
    assert all(r["document_relevant"] is None for r in unsupported["ranked_results"])
    original = output.read_bytes()

    def unexpected_run(*args, **kwargs):
        pytest.fail("An existing report must be rejected before running evaluation")

    monkeypatch.setattr(evaluation, "run_evaluation", unexpected_run)
    with pytest.raises(SystemExit) as exc:
        evaluation.main(args)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "Report already exists" in error
    assert "OUTPUT=" in error and "--output" in error
    assert "Traceback" not in error
    assert output.read_bytes() == original


def test_stable_matching_when_chunk_boundaries_change(dataset_path, fake_embeddings):
    baseline = evaluation.run_evaluation(dataset_path, top_k=20)
    changed = evaluation.run_evaluation(dataset_path, top_k=20, chunk_size=90, overlap=10)
    assert baseline["counts"]["chunks"] != changed["counts"]["chunks"]
    for report in (baseline, changed):
        assert report["error"] is None
        assert report["questions"][1]["metrics"]["20"]["recall"] == 1
        assert {r["document_id"] for r in report["questions"][1]["ranked_results"]} == {
            "alpha",
            "beta",
        }
    assert baseline["dataset"]["sha256"] == changed["dataset"]["sha256"]
    assert baseline["configuration_sha256"] != changed["configuration_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(documents=[]),
        lambda d: d.update(questions=[]),
        lambda d: d.update(version=True),
        lambda d: d["documents"].append(copy.deepcopy(d["documents"][0])),
        lambda d: d["documents"][1].update(path="alpha.md"),
        lambda d: d["documents"][0].update(path="../outside.md"),
        lambda d: d["questions"].append(copy.deepcopy(d["questions"][0])),
        lambda d: d["questions"][0].update(question="  "),
        lambda d: d["questions"][0].pop("unsupported"),
        lambda d: d["questions"][0].update(evidence=[]),
        lambda d: d["questions"][0]["evidence"][0].update(document_id="unknown"),
        lambda d: d["questions"][0]["evidence"][0].update(document_id=[]),
        lambda d: d["questions"][0]["evidence"][0].update(quote="not actually in source"),
        lambda d: d["questions"][2].pop("reason"),
        lambda d: d["questions"][2].update(
            evidence=[{"document_id": "alpha", "quote": "Blue boat."}]
        ),
    ],
)
def test_malformed_datasets(dataset_path, mutation):
    data = json.loads(dataset_path.read_text())
    mutation(data)
    dataset_path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evaluation.load_dataset(dataset_path)


@pytest.mark.parametrize("content", ["", "{", "[]", "null", "{}"])
def test_empty_or_invalid_json(dataset_path, content):
    dataset_path.write_text(content)
    with pytest.raises(ValueError):
        evaluation.load_dataset(dataset_path)


def test_checked_in_dataset_and_fingerprints(dataset_path):
    dataset = evaluation.load_dataset(evaluation.DEFAULT_DATASET)
    assert len(dataset.paths) == 18
    assert len(dataset.manifest["questions"]) == 26
    assert sum(q["unsupported"] for q in dataset.manifest["questions"]) == 3
    original = evaluation.load_dataset(dataset_path).fingerprint
    assert evaluation.load_dataset(dataset_path).fingerprint == original
    (dataset_path.parent / "alpha.md").write_text("Alpha station. Blue boat. Edited.")
    assert evaluation.load_dataset(dataset_path).fingerprint != original


def test_ingestion_failure_records_skipped_questions(dataset_path, monkeypatch):
    def fail(texts):
        raise OSError("No cached model")

    monkeypatch.setattr(evaluation, "RetrievalPipeline", lambda: RetrievalPipeline(fail))
    report = evaluation.run_evaluation(dataset_path)
    assert "No cached model" in report["error"]
    assert report["aggregate"]["scored_count"] == 0
    assert report["aggregate"]["failed_or_skipped_supported_count"] == 2
    assert all(q["status"] == "skipped" and q["error"] for q in report["questions"])
    assert report["aggregate"]["by_k"]["5"]["recall"] is None


def test_query_failure_retains_other_questions(
    dataset_path, fake_embeddings, monkeypatch, tmp_path
):
    original = RetrievalPipeline.embed_query

    def embed(self, query):
        if query == "Where is the blue boat?":
            raise RuntimeError("Query failed")
        return original(self, query)

    monkeypatch.setattr(RetrievalPipeline, "embed_query", embed)
    output = tmp_path / "failed.json"
    assert (
        evaluation.main(["--dataset", str(dataset_path), "--output", str(output), "--top-k", "2"])
        == 1
    )
    report = json.loads(output.read_text())
    assert report["questions"][0]["error"] == "Query failed"
    assert report["questions"][1]["status"] == "completed"
    assert report["aggregate"]["scored_count"] == 1
    assert set(report["aggregate"]["by_k"]) == {"1", "2"}


def test_dataset_error_writes_report(tmp_path):
    output = tmp_path / "invalid.json"
    assert (
        evaluation.main(["--dataset", str(tmp_path / "absent.json"), "--output", str(output)]) == 1
    )
    report = json.loads(output.read_text())
    assert report["error"]
    assert report["aggregate"]["scored_count"] == 0
