"""Insights depend only on measured values, never inference or a network."""

from dataclasses import replace
from pathlib import Path

import pytest

from rag_lab.chunking import Chunk, TokenDiagnostics
from rag_lab.insights import (
    IndexInsights,
    overlap_fraction,
    render_index_insights,
    render_search_insights,
    render_summary_metrics,
    suggest_experiment,
)
from rag_lab.pipeline import RetrievalPipeline
from rag_lab.retrieval import SearchResult


def chunk(start=0, end=20, document=1, diagnostics=None):
    return Chunk(
        f"doc-{document:04d}-chunk-{start:04d}",
        "x" * (end - start),
        "same.md",
        None,
        start,
        end,
        diagnostics,
    )


def test_scores_and_interpretations():
    results = [SearchResult(chunk(), 0.4138), SearchResult(chunk(10, 30), 0.4012)]
    text = render_search_insights(results)
    assert "0.4138" in text
    assert "0.0126" in text
    assert "50.0%" in text
    assert "not a calibrated confidence" in text
    assert "does not establish equal relevance" in text
    assert "display heuristic" in text
    assert text == render_search_insights(results)
    assert "41.38%" not in text


def test_zero_one_and_incomplete_diagnostics():
    assert "No passages retrieved." in render_search_insights([])
    text = render_search_insights([SearchResult(chunk(), -0.2)])
    assert "-0.2000" in text
    assert "Top-two" not in text
    assert "unavailable" in text
    assert "<dd>Unavailable</dd>" in text
    assert "exceed the embedding" not in text
    assert "Largest source overlap" not in text


def test_truncation_only_when_measured():
    short = chunk(diagnostics=TokenDiagnostics(5, 5, 5))
    long = chunk(10, 30, diagnostics=TokenDiagnostics(8, 5, 5))
    text = render_search_insights([SearchResult(short, 0.8)])
    assert "exceed the embedding" not in text
    text = render_search_insights([SearchResult(long, 0.8), SearchResult(short, 0.2)])
    assert "1 retrieved passage exceeds" in text
    assert "Top-two margin is small" not in text
    assert "Token diagnostics are unavailable" not in text
    index = IndexInsights.from_chunks([short, long, chunk()], 1, "Sentence-aware", 20, 5, "Fake")
    text = render_index_insights(index)
    assert "1 of 2 diagnosed chunks" in text
    assert "unavailable" in text
    assert "Effective input limit: 5" in text
    assert "average 20.0, minimum 20, maximum 20" in text


def test_overlap_requires_source_identity():
    assert overlap_fraction(chunk(), chunk(10, 30)) == 0.5
    assert overlap_fraction(chunk(), chunk(20, 40)) == 0
    assert overlap_fraction(chunk(), chunk(document=2)) is None
    assert overlap_fraction(chunk(), replace(chunk(), page=1)) is None
    assert overlap_fraction(chunk(), replace(chunk(), identifier="unknown")) is None


@pytest.mark.parametrize("strategy", ["Character-based", "Sentence-aware"])
def test_sample_index_search_and_cached_diagnostics(strategy):
    calls = []

    def diagnose(texts):
        calls.append(texts)
        return [TokenDiagnostics(12, 10, 10) for _ in texts]

    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts], diagnose)
    sample = Path(__file__).parents[1] / "samples" / "example.md"
    count = pipeline.index_documents([sample], 300, 40, strategy)
    assert count > 1
    results = pipeline.search("How do I measure retrieval quality?")
    pipeline.search("Same query again")
    assert len(calls) == 1
    assert all(item.chunk.diagnostics.truncated for item in results)
    assert all(item.chunk.filename == "example.md" for item in results)
    assert pipeline.insights.strategy == strategy
    assert pipeline.insights.truncated == count
    assert "What should I try next?" in render_index_insights(pipeline.insights)


def test_sections_diagnostics_escape_and_single_experiment():
    item = chunk(diagnostics=TokenDiagnostics(12, 10, 10))
    index = IndexInsights.from_chunks([item], 1, "<script>strategy</script>", 20, 5, "<b>model</b>")
    html = render_index_insights(index)
    for heading in ("What happened?", "What does it mean?", "What should I try next?"):
        assert f"<h3>{heading}</h3>" in html
    assert '<details class="index-diagnostics">' in html
    assert "<summary>Indexing diagnostics</summary>" in html
    assert "<script>" not in html
    assert "<b>model</b>" not in html
    assert "&lt;b&gt;model&lt;/b&gt;" in html
    results = [SearchResult(item, 0.5), SearchResult(chunk(5, 25), 0.49)]
    index = IndexInsights.from_chunks([item], 1, "Sentence-aware", 800, 120, "Fake")
    html = render_search_insights(results, index)
    experiment = html.split("<h3>What should I try next?</h3>", 1)[1]
    assert "reducing the chunk size from 800 to 500" in experiment
    assert "reducing overlap" not in experiment
    assert "top two" not in experiment
    assert html == render_search_insights(results, index)


@pytest.mark.parametrize("score,small", [(0.49, True), (0.45, False)])
def test_small_margin_suggestion_without_overlap(score, small):
    index = IndexInsights.from_chunks([chunk()], 1, "Sentence-aware", 800, 120, "Fake")
    html = render_search_insights(
        [
            SearchResult(chunk(document=1), 0.5),
            SearchResult(chunk(document=2), score),
        ],
        index,
    )
    assert ("almost identical" in html) == small
    experiment = html.split("<h3>What should I try next?</h3>", 1)[1]
    assert ("from 800 to 500 characters" in experiment) == small


def test_summary_metrics_use_index_wide_diagnostics():
    short = chunk(diagnostics=TokenDiagnostics(5, 5, 5))
    long = chunk(10, 30, diagnostics=TokenDiagnostics(8, 5, 5))
    index = IndexInsights.from_chunks(
        [short, long, chunk()], 1, "Character-based", 800, 120, "Fake"
    )
    results = [SearchResult(short, 0.41385), SearchResult(short, 0.4012)]
    html = render_summary_metrics(results, index)
    assert "<dt>Top similarity</dt><dd>0.4138</dd>" in html
    assert "<dt>Score margin</dt><dd>0.0126</dd>" in html
    assert "Token truncation <small>Index-wide</small>" in html
    assert "<dd>1 / 2 diagnosed</dd>" in html
    assert html.count("<div>") == 3
    assert "%" not in html
    # Truncated indexed chunks need not appear in the retrieved results.
    rendered = render_search_insights([SearchResult(short, 0.8)], index)
    assert "1 / 2 diagnosed" in rendered
    assert "count covers 2 of 3 chunks" in rendered
    assert "retrieved passages exceed" not in rendered
    assert "whether truncation decreases" in rendered


@pytest.mark.parametrize("results", [[], [SearchResult(chunk(), -0.2)]])
def test_missing_margin_and_index_diagnostics(results):
    html = render_summary_metrics(results)
    assert "<dt>Score margin</dt><dd>Unavailable</dd>" in html
    assert "<small>Index-wide</small></dt><dd>Unavailable</dd>" in html
    assert ("<dt>Top similarity</dt><dd>Unavailable</dd>" in html) == (not results)


def test_empty_results_keep_index_summary_and_no_search_scores():
    item = chunk(diagnostics=TokenDiagnostics(5, 5, 5))
    index = IndexInsights.from_chunks([item], 1, "Character-based", 800, 120, "Fake")
    html = render_search_insights([], index)
    assert "No passages retrieved" in html
    assert "0 / 1 diagnosed" in html
    assert "<dt>Top similarity</dt><dd>Unavailable</dd>" in html
    assert "<dt>Score margin</dt><dd>Unavailable</dd>" in html
    assert "There are no passages to compare" in html


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (size, overlap)
        for size in (100, 150, 500, 550, 800, 2000)
        for overlap in (0, 90, 120, 490, 500)
        if overlap < size
    ],
)
def test_smaller_chunk_experiments_stay_within_slider_bounds(size, overlap):
    import re

    from rag_lab.chunking import validate_parameters

    item = chunk(diagnostics=TokenDiagnostics(8, 5, 5))
    index = IndexInsights.from_chunks([item], 1, "Sentence-aware", size, overlap, "Fake")
    text = suggest_experiment(index)
    if size == 100:
        assert "No smaller-chunk experiment" in text
        assert "Try" not in text
        return
    match = re.search(r"chunk size from (\d+) to (\d+)", text)
    assert match
    original, proposed_size = map(int, match.groups())
    assert original == size
    assert 100 <= proposed_size < size <= 2000
    assert proposed_size % 50 == 0
    overlap_match = re.search(r"overlap from (\d+) to (\d+)", text)
    proposed_overlap = int(overlap_match[2]) if overlap_match else overlap
    assert 0 <= proposed_overlap <= 500
    assert proposed_overlap % 10 == 0
    validate_parameters(proposed_size, proposed_overlap)
    assert "whether truncation decreases" in text
    assert index.chunk_size == size and index.overlap == overlap


def test_experiment_priorities_and_minimum_chunk_size():
    index = IndexInsights.from_chunks([chunk()], 1, "Character-based", 800, 120, "Fake")
    assert "sentence-aware chunking with the same size and overlap" in suggest_experiment(index)
    text = suggest_experiment(index, close_scores=True)
    assert "from 800 to 500 characters" in text
    assert "whether the ranking changes" in text
    assert "from 120 to 60 characters" in suggest_experiment(index, repeated_source=True)
    assert "overlap" in suggest_experiment(index, close_scores=True, repeated_source=True)
    sentence = replace(index, strategy="Sentence-aware", chunk_size=100, overlap=0)
    assert "No further configuration experiment" in suggest_experiment(sentence, close_scores=True)
    assert "No further configuration experiment" in suggest_experiment(
        sentence, repeated_source=True
    )
    character = replace(sentence, strategy="Character-based")
    assert "sentence-aware" in suggest_experiment(character, close_scores=True)
    assert "configuration is unavailable" in suggest_experiment(None)
