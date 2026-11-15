"""Smoke-report infrastructure uses the real pipeline with fake model boundaries."""

import json
import os

import pytest

from rag_lab import smoke
from rag_lab.generation import ABSTENTION, LocalGenerator
from rag_lab.pipeline import RetrievalPipeline


@pytest.fixture
def models(monkeypatch):
    def encode(texts):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        return [[1, 0] for _ in texts]

    monkeypatch.setattr(smoke, "RetrievalPipeline", lambda: RetrievalPipeline(encode))

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(len(" ".join(message["content"] for message in messages).split())))

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return text.split()

    def load(self):
        self.tokenizer = Tokenizer()
        self.context_window = 4096

    answers = iter(
        (
            "Lark, starting 14 June 2031. [S1]",
            "Mira Vale. [S1] The budget is not provided. [S99]",
            ABSTENTION,
        )
    )
    monkeypatch.setattr(LocalGenerator, "load", load)
    monkeypatch.setattr(LocalGenerator, "generate", lambda self, prepared: next(answers))


def test_report_preserves_evidence_diagnostics_and_pending_judgements(models):
    report = smoke.run_smoke("/fake/model")
    # Check the artifact itself is JSON serializable, not just Python dataclasses.
    report = json.loads(json.dumps(report))
    assert report["error"] is None
    assert report["chunk_count"] == 2
    assert report["ingestion_seconds"] >= 0
    assert report["settings"]["embedding_model"] == smoke.DEFAULT_MODEL
    assert len(report["fixture"]["harbour.txt"]["sha256"]) == 64
    assert [row["name"] for row in report["scenarios"]] == [row[0] for row in smoke.SCENARIOS]
    for row in report["scenarios"]:
        assert row["question"] and row["expected_behavior_for_human_review"]
        assert row["human_review"]["claims_supported_by_assembled_context"] is None
        assert row["human_review"]["meets_expected_behavior"] is None
        automated = row["automated"]
        assert automated["inference_completed"]
        assert automated["answer"]
        assert len(automated["retrieved_sources"]) == 2
        context = automated["context"]
        for source in context["sources"]:
            assert source["result"]["chunk"]["text"] in context["text"]
        assert context["omitted_ids"] == []
        assert automated["diagnostics"]["prompt_tokens"] > 0
        assert automated["output_tokens_reencoded_estimate"] > 0
        assert all(value >= 0 for value in automated["diagnostics"]["durations"].values())
    partial = report["scenarios"][1]["automated"]["citation_label_validation"]
    assert partial["labels_used"] == ["S1", "S99"]
    assert partial["valid_labels"] == ["S1"]
    assert partial["warnings"] == ["Unknown source reference: [S99]."]
    assert report["scenarios"][2]["automated"]["status"] == "insufficient_context"


def test_main_writes_new_report_and_refuses_overwrite(models, tmp_path):
    output = tmp_path / "reports" / "run.json"
    args = ["--model-dir", "/fake/model", "--output", str(output)]
    assert smoke.main(args) == 0
    original = output.read_bytes()
    assert len(json.loads(original)["scenarios"]) == 3
    with pytest.raises(FileExistsError):
        smoke.main(args)
    assert output.read_bytes() == original


def test_generation_failure_is_recorded_for_all_scenarios(models, monkeypatch, tmp_path):
    def fail(self, prepared):
        raise RuntimeError("Simulated inference failure")

    monkeypatch.setattr(LocalGenerator, "generate", fail)
    output = tmp_path / "failed.json"
    assert smoke.main(["--model-dir", "/fake/model", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    for row in report["scenarios"]:
        automated = row["automated"]
        assert not automated["inference_completed"]
        assert "Simulated inference failure" in automated["message"]
        assert automated["context"]["text"]
        assert automated["answer"] is None
        assert automated["output_tokens_reencoded_estimate"] is None


def test_uncached_embeddings_fail_without_downloading(monkeypatch, tmp_path):
    def fail(texts):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        raise OSError("Not cached")

    monkeypatch.setattr(smoke, "RetrievalPipeline", lambda: RetrievalPipeline(fail))
    output = tmp_path / "uncached.json"
    assert smoke.main(["--model-dir", "/fake/model", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert "cached embeddings required" in report["error"]
    assert report["ingestion_seconds"] >= 0
    assert report["scenarios"] == []


def test_query_error_does_not_discard_other_scenarios(models, monkeypatch):
    original = RetrievalPipeline.embed_query

    def embed(self, question):
        if question == smoke.SCENARIOS[0][1]:
            raise RuntimeError("Simulated query failure")
        return original(self, question)

    monkeypatch.setattr(RetrievalPipeline, "embed_query", embed)
    report = smoke.run_smoke("/fake/model")
    assert report["scenarios"][0]["error"] == "Simulated query failure"
    assert report["scenarios"][0]["automated"] is None
    assert report["scenarios"][1]["automated"]["inference_completed"]
    assert report["scenarios"][2]["automated"]["inference_completed"]
