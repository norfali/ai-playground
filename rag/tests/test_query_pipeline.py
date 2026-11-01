"""Explicit query composition with fake embeddings, tokenizer and LLM responses."""

import pytest

from rag_lab import generation
from rag_lab.chunking import Chunk
from rag_lab.context import assemble_context
from rag_lab.generation import ABSTENTION, LocalGenerator, check_citations
from rag_lab.pipeline import IngestionPipeline, QueryPipeline, RetrievalPipeline
from rag_lab.retrieval import SearchResult


@pytest.fixture
def retrieval(tmp_path):
    path = tmp_path / "evidence.txt"
    path.write_text("Cats nap. Dogs run.")
    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    pipeline.index_documents([path], chunk_size=10, overlap=0)
    return pipeline


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        # Deliberately artificial tokenizer: special tokens plus three per word.
        return [101, 102] + [7] * (3 * len(" ".join(m["content"] for m in messages).split()))


@pytest.fixture
def generator(monkeypatch):
    model = LocalGenerator("/fake/model", max_answer_tokens=50)
    model.tokenizer = FakeTokenizer()
    model.context_window = 500
    monkeypatch.setattr(model, "load", lambda: None)
    monkeypatch.setattr(model, "generate", lambda prepared: "Cats nap. [S1]")
    return model


def result(identifier="one", text="Exact <source> text.\n", page=3, score=0.8):
    return SearchResult(Chunk(identifier, text, "evidence.pdf", page, 12, 12 + len(text)), score)


def test_context_preserves_exact_text_offsets_and_rank_labels():
    results = [result(), result("two", " " * 2), result("three", "Other passage", None)]
    context = assemble_context(results)
    assert [source.label for source in context.sources] == ["S1", "S3"]
    assert context.omitted_ids == ("two",)
    assert results[0].chunk.text in context.text
    assert "page 3" in context.text and "characters 12:33" in context.text
    assert "[S3] evidence.pdf | three" in context.text
    assert context.sources[0].result is results[0]
    assert assemble_context([]).text == ""


def test_ingestion_and_query_data_flow_and_timings(retrieval):
    assert isinstance(retrieval, IngestionPipeline)
    calls = []
    retrieval.encoder = lambda texts: calls.append(texts) or [[1, 0]]
    ticks = iter(range(20))
    response = QueryPipeline(retrieval, clock=lambda: float(next(ticks))).run("Cats?", 1)
    assert calls == [["Cats?"]]
    assert len(response.results) == len(response.context.sources) == 1
    assert response.context.sources[0].result is response.results[0]
    assert response.answer is None
    assert response.status == "retrieval_only"
    diagnostics = response.diagnostics
    assert diagnostics.model is None and diagnostics.prompt_tokens is None
    assert diagnostics.durations == {
        "query_embedding": 1.0,
        "retrieval": 1.0,
        "context_assembly": 1.0,
        "model_loading": None,
        "generation": None,
        "total": 7.0,
    }
    assert diagnostics.retrieved == ((response.results[0].chunk.identifier, 1.0),)


def test_mock_grounded_answer(retrieval, generator):
    response = QueryPipeline(retrieval, generator).run("What do cats do?")
    assert response.answer == "Cats nap. [S1]"
    assert response.status == "generated"
    assert response.citations == ("S1",)
    assert response.warnings == ()
    assert response.diagnostics.model == "/fake/model"
    assert response.diagnostics.prompt_tokens > 0
    assert response.diagnostics.context_window == 500
    assert response.diagnostics.max_answer_tokens == 50
    durations = response.diagnostics.durations
    assert all(duration >= 0 for duration in durations.values())
    assert durations["total"] >= sum(value for key, value in durations.items() if key != "total")


def test_tokenizer_budget_includes_template_question_and_answer(generator):
    results = [result("long", "word " * 500), result("short", "A short passage")]
    prepared = generator.prepare("Question?", results)
    assert prepared.context.omitted_ids == ("long",)
    assert prepared.context.sources[0].label == "S2"
    assert prepared.tokens == generator.tokenizer.apply_chat_template(
        generation.prompt_messages("Question?", prepared.context.text),
        tokenize=True,
        add_generation_prompt=True,
    )
    assert len(prepared.tokens) + prepared.max_answer_tokens <= prepared.context_window
    with pytest.raises(ValueError, match="Question and instructions"):
        generator.prepare("question " * 500, results)


@pytest.mark.parametrize(
    "directory", [None, "", "  ", "/missing/model", "mlx-community/not-a-local-path"]
)
def test_unavailable_model_preserves_retrieval(retrieval, directory):
    response = QueryPipeline(retrieval, LocalGenerator(directory)).run("question")
    assert response.status == "generation_unavailable"
    assert response.results and response.context.sources
    assert response.answer is None
    assert response.diagnostics.durations["model_loading"] >= 0
    assert response.diagnostics.durations["generation"] is None


def test_no_context_skips_model_loading(retrieval, generator, monkeypatch):
    monkeypatch.setattr(retrieval, "retrieve", lambda vector, top_k: [])
    monkeypatch.setattr(generator, "load", lambda: pytest.fail("No evidence; must not load LLM"))
    response = QueryPipeline(retrieval, generator).run("question")
    assert response.status == "insufficient_context" and response.answer == ABSTENTION
    assert response.diagnostics.durations["model_loading"] is None


def test_no_passage_fits_skips_generation(retrieval, generator, monkeypatch):
    base_tokens = len(
        generator.tokenizer.apply_chat_template(
            generation.prompt_messages("question", ""), tokenize=True, add_generation_prompt=True
        )
    )
    generator.context_window = base_tokens + generator.max_answer_tokens
    monkeypatch.setattr(generator, "generate", lambda prepared: pytest.fail("No context fits"))
    response = QueryPipeline(retrieval, generator).run("question")
    assert response.status == "insufficient_context"
    assert response.context.sources == ()
    assert response.context.omitted_ids
    assert response.diagnostics.durations["generation"] is None


@pytest.mark.parametrize("answer", [ABSTENTION, "Uncited claim", "Claim [S99]", ""])
def test_abstention_citation_warnings_and_empty_output(retrieval, generator, monkeypatch, answer):
    monkeypatch.setattr(generator, "generate", lambda prepared: answer)
    response = QueryPipeline(retrieval, generator).run("question")
    if answer == ABSTENTION:
        assert response.status == "insufficient_context" and response.warnings == ()
    elif not answer:
        assert response.status == "generation_unavailable" and response.answer is None
    else:
        assert response.warnings and response.citations == ()
    assert response.results


def test_generation_failure_has_timing_and_evidence(retrieval, generator, monkeypatch):
    def fail(prepared):
        raise RuntimeError("Out of memory")

    monkeypatch.setattr(generator, "generate", fail)
    response = QueryPipeline(retrieval, generator).run("question")
    assert "Out of memory" in response.message
    assert response.results and response.context.sources
    assert response.diagnostics.durations["generation"] >= 0


def test_unsupported_chat_template_preserves_evidence(retrieval, generator, monkeypatch):
    from jinja2 import TemplateError

    def fail(*args, **kwargs):
        raise TemplateError("System role not supported")

    monkeypatch.setattr(generator.tokenizer, "apply_chat_template", fail)
    response = QueryPipeline(retrieval, generator).run("question")
    assert response.status == "generation_unavailable"
    assert "chat template" in response.message
    assert response.context.sources
    assert response.diagnostics.durations["generation"] is None


def test_citations_are_membership_checks_only():
    context = assemble_context([result()])
    valid, warnings = check_citations("Claim [S1] repeated [S1] and [S8]", context)
    assert valid == ("S1",)
    assert warnings == ("Unknown source reference: [S8].",)
