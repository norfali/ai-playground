"""Callbacks, escaped content and session state, without running inference."""

import asyncio

import pytest
from gradio.state_holder import SessionState

from rag_lab import app
from rag_lab.chunking import Chunk
from rag_lab.pipeline import RetrievalPipeline
from rag_lab.retrieval import SearchResult


def test_results_escape_untrusted_content():
    chunk = Chunk("<b>id</b>", '<script>alert("x")</script>', "<img src=x>.pdf", 3, 0, 10)
    html = app.render_results([SearchResult(chunk, -0.12345)])
    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;id&lt;/b&gt;" in html
    assert "Page 3" in html
    assert "Cosine similarity:" in html
    assert "-0.1235" in html


def test_callbacks_and_session_isolation(tmp_path, monkeypatch):
    path = tmp_path / "notes.md"
    path.write_text("Useful source text")
    monkeypatch.setattr(
        app, "RetrievalPipeline", lambda: RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    )
    first, status, results = app.index_documents([str(path)], 800, 120)
    second, _, _ = app.index_documents([str(path)], 800, 120)
    assert first is not second
    assert first.index is not second.index
    assert "1 documents and 1 chunks" in status
    assert results == ""
    status, results = app.search_documents(first, "text", 5)
    assert "Retrieved 1 passages" in status
    assert "Useful source text" in results
    state, status, results = app.index_documents([], 800, 120)
    assert state is None
    assert "No collection is indexed" in status
    assert results == ""


def test_clear_errors():
    assert app.search_documents(None, "query", 5) == ("Index documents before searching.", "")
    status, results = app.search_documents(RetrievalPipeline(), "", 5)
    assert "non-empty" in status
    assert results == ""
    assert "No passages" in app.render_results([])


def test_build_app():
    demo = app.build_app()
    assert demo.analytics_enabled is False
    assert demo.title == "RagLab"
    states = [block for block in demo.blocks.values() if isinstance(block, app.gr.State)]
    assert len(states) == 1
    assert states[0].value is None
    assert states[0].time_to_live == 3600
    sliders = [block for block in demo.blocks.values() if isinstance(block, app.gr.Slider)]
    assert len(sliders) == 3
    assert all(isinstance(slider.preprocess(float(slider.value)), int) for slider in sliders)


def test_insights_callbacks_clear_stale_search(tmp_path, monkeypatch):
    from rag_lab.chunking import TokenDiagnostics

    path = tmp_path / "source.md"
    path.write_text("One sentence. Another sentence. Final sentence.")
    monkeypatch.setattr(
        app,
        "RetrievalPipeline",
        lambda: RetrievalPipeline(
            lambda texts: [[1, 0] for _ in texts],
            lambda texts: [TokenDiagnostics(12, 10, 10) for _ in texts],
        ),
    )
    pipeline, _, results, insights = app.index_with_insights([str(path)], 25, 5, "Sentence-aware")
    assert results == ""
    assert "Sentence-aware" in insights
    assert "Search measurements" not in insights
    _, results, insights = app.search_with_insights(pipeline, "sentence", 2)
    assert "Search measurements" in insights
    assert "12 original" in results
    assert "10 processed" in results
    assert "Truncated" in results
    status, results, insights = app.search_with_insights(pipeline, "", 2)
    assert "Search failed" in status
    assert results == ""
    assert "Search measurements" not in insights
    pipeline, _, results, insights = app.index_with_insights([], 25, 5, "Sentence-aware")
    assert pipeline is None
    assert results == insights == ""


def test_insights_panel_and_selector_are_native_components():
    demo = app.build_app()
    radios = [block for block in demo.blocks.values() if isinstance(block, app.gr.Radio)]
    assert len(radios) == 1
    assert radios[0].value == "Character-based"
    markdown = [block.value for block in demo.blocks.values() if isinstance(block, app.gr.Markdown)]
    assert "## Retrieval Insights" in markdown


def test_full_passages_and_explicit_token_states():
    from html import unescape

    from rag_lab.chunking import TokenDiagnostics

    text = "a" * 4000 + "\n\n<b>full & intact</b>\n  trailing spaces  "
    chunk = Chunk("doc-0001-chunk-0001", text, "source.md", None, 0, len(text))
    html = app.render_results([SearchResult(chunk, 0.25)])
    assert text in unescape(html)
    assert "<pre" not in html
    assert "Token diagnostics unavailable" in html
    assert "Page" not in html
    assert "truncation-warning" not in html
    diagnosed = app.render_results(
        [
            SearchResult(
                Chunk("id", text, "source.md", 1, 0, len(text), TokenDiagnostics(10, 10, 20)),
                0.25,
            )
        ]
    )
    assert "10 / 20 tokens · No truncation" in diagnosed
    assert "Within embedding token limit" not in diagnosed
    assert "10 original" in diagnosed
    assert "10 processed" in diagnosed


def test_clear_search_retains_current_index_measurements(tmp_path, monkeypatch):
    path = tmp_path / "notes.md"
    path.write_text("An indexed passage.")
    monkeypatch.setattr(
        app, "RetrievalPipeline", lambda: RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    )
    pipeline, _, _, _ = app.index_with_insights([str(path)], 800, 120, "Character-based")
    status, results, insights = app.clear_search(pipeline)
    assert "Select Search" in status
    assert results == ""
    assert "1 documents · 1 chunks" in insights
    assert "Search measurements" not in insights
    assert app.clear_search(None)[1:] == ("", "")


def test_workspace_components_and_invalidation_wiring():
    demo = app.build_app()
    panels = [
        block
        for block in demo.blocks.values()
        if isinstance(block, app.gr.HTML)
        and block.elem_id in ("retrieved-passages", "retrieval-insights")
    ]
    assert len(panels) == 2
    assert all('tabindex="0"' in panel.html_template for panel in panels)
    assert all('role="region"' in panel.html_template for panel in panels)
    callbacks = list(demo.fns.values())
    clear_events = [fn for fn in callbacks if fn.fn is app.clear_query]
    assert len(clear_events) == 4
    assert all(
        [output.elem_id for output in fn.outputs[1:3]]
        == ["retrieved-passages", "retrieval-insights"]
        for fn in clear_events
    )
    search_events = [fn for fn in callbacks if fn.fn is app.query_with_pipeline]
    assert len(search_events) == 2  # Search button and Enter in the query field.
    assert all(fn.concurrency_id == "retrieval" for fn in clear_events + search_events)
    assert all(len(fn.inputs) == 5 for fn in search_events)
    assert all(fn.js is None for fn in search_events)


def test_repeated_searches_through_gradio_preserve_collection_and_settings(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Semantic search matches meaning. Keyword search matches words.")
    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    pipeline.index_documents([path])
    demo = app.build_app()
    searches = [fn for fn in demo.fns.values() if fn.fn is app.query_with_pipeline]
    clear = next(fn for fn in demo.fns.values() if fn.fn is app.clear_query)
    state = SessionState(demo)
    state[searches[0].inputs[0]._id] = pipeline

    async def search_twice():
        for handler, question in zip(
            searches,
            [
                "How does semantic search differ from keyword search?",
                "What is Australia's capital?",
            ],
            strict=True,
        ):
            # User edits clear previous outputs, then Search or Enter submits
            # all five declared inputs. State is resolved by Gradio on the server.
            await demo.process_api(clear, [None], state=state)
            response = await demo.process_api(handler, [None, question, 5, False, ""], state=state)
            assert "Retrieved 1 passages" in response["data"][0]
            assert "Semantic search matches meaning" in response["data"][1]
            assert response["data"][6].root["status"] == "retrieval_only"
            assert state[handler.inputs[0]._id] is pipeline

    asyncio.run(search_twice())


def test_pipeline_callback_retrieval_and_failure_preserve_evidence(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("A source passage with <script>markup</script>.")
    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    pipeline.index_documents([path])
    response = app.query_with_pipeline(pipeline, "question", 5, False, "/missing")
    assert "generation skipped" in response[0]
    assert response[3] == ""
    assert "Included in context" in response[4]
    assert "[S1] notes.txt" in response[5]
    assert response[6]["durations"]["generation"] is None
    failed = app.query_with_pipeline(pipeline, "question", 5, True, "/missing")
    assert "unavailable" in failed[0]
    assert "source passage" in failed[1]
    assert failed[5] == response[5]
    assert failed[6]["status"] == "generation_unavailable"
    assert app.query_with_pipeline(None, "q", 5, False, "")[1:] == ("", "", "", "", "", {})
    assert app.query_with_pipeline(pipeline, "", 5, False, "")[3:] == ("", "", "", {})
    assert app.clear_query(pipeline)[3:] == ("", "", "", {})


@pytest.mark.parametrize("generate_answer", [False, True])
def test_unset_model_directory_preserves_retrieval(tmp_path, generate_answer):
    path = tmp_path / "notes.txt"
    path.write_text("Semantic search matches meaning. Keyword search matches words.")
    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    pipeline.index_documents([path])
    response = app.query_with_pipeline(
        pipeline,
        "What are the main differences between semantic search and keyword search, "
        "and when would you choose one over the other?",
        5,
        generate_answer,
        None,
    )
    assert "Semantic search matches meaning" in response[1]
    assert response[3] == ""
    assert "[S1] notes.txt" in response[5]
    assert response[6]["durations"]["generation"] is None
    if generate_answer:
        assert "Choose an existing local model directory" in response[0]
        assert response[6]["status"] == "generation_unavailable"
    else:
        assert response[6]["status"] == "retrieval_only"


def test_answer_source_references_escape_filenames_and_distinguish_citations(tmp_path):
    from dataclasses import replace

    from rag_lab.pipeline import QueryPipeline

    path = tmp_path / "<img src=x>.txt"
    path.write_text("Source evidence.")
    pipeline = RetrievalPipeline(lambda texts: [[1, 0] for _ in texts])
    pipeline.index_documents([path])
    response = QueryPipeline(pipeline).run("question")
    response = replace(response, citations=("S1",), warnings=("Unknown [S9] <script>",))
    sources = app.render_sources(response)
    assert "<img" not in sources and "<script>" not in sources
    assert "&lt;img" in sources and "&lt;script&gt;" in sources
    assert "Cited by model" in sources
    assert "doc-0001-chunk-0001" in sources
    assert "Included in context" in app.render_sources(replace(response, citations=()))


def test_github_header_links_to_verified_repository_directory():
    demo = app.build_app()
    header = next(
        block.value
        for block in demo.blocks.values()
        if isinstance(block, app.gr.HTML) and 'class="project-header"' in block.value
    )
    assert 'href="https://github.com/norfali/ai-playground/tree/main/rag"' in header
    assert 'target="_blank"' in header
    assert 'rel="noopener noreferrer"' in header
    assert 'aria-label="View RagLab on GitHub (opens in a new tab)"' in header
    assert "View on GitHub" in header
    assert "Local retrieval playground" not in header


def test_compact_result_token_lines():
    from rag_lab.chunking import TokenDiagnostics

    short = Chunk(
        "short", "Full short passage", "source.md", None, 0, 18, TokenDiagnostics(115, 115, 256)
    )
    long = Chunk(
        "long", "Full long passage", "source.md", None, 0, 17, TokenDiagnostics(310, 256, 256)
    )
    html = app.render_results([SearchResult(short, 0.8), SearchResult(long, 0.7)])
    assert "115 / 256 tokens · No truncation" in html
    assert "115 original tokens, 115 processed tokens" in html
    assert "310 original · 256 processed · Truncated (limit 256 tokens)" in html
    assert html.count('class="truncation-warning"') == 1
    assert "Within embedding token limit" not in html
    assert "Full short passage" in html and "Full long passage" in html
