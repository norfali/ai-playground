"""Thin Gradio callbacks and session-local retrieval state."""

from html import escape
from pathlib import Path

import gradio as gr
from gradio.themes import Base

from rag_lab.chunking import STRATEGIES
from rag_lab.insights import (
    render_index_diagnostics,
    render_index_insights,
    render_search_insights,
)
from rag_lab.pipeline import RetrievalPipeline
from rag_lab.retrieval import SearchResult

STYLES = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")


def build_theme() -> Base:
    """Use Gradio's mode-aware tokens and local system fonts; no font requests."""
    return Base(
        primary_hue="teal",
        secondary_hue="violet",
        neutral_hue="slate",
        font=["ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "monospace"],
        radius_size="md",
    ).set(
        body_background_fill="#fafaf7",
        body_background_fill_dark="#111b21",
        body_text_color="*neutral_800",
        body_text_color_dark="*neutral_100",
        body_text_color_subdued="*neutral_600",
        body_text_color_subdued_dark="*neutral_300",
        block_label_text_color="*neutral_600",
        block_label_text_color_dark="*neutral_300",
        input_placeholder_color="*neutral_500",
        input_placeholder_color_dark="*neutral_400",
        input_background_fill="white",
        input_background_fill_dark="*neutral_900",
        input_border_width="1px",
        input_border_color="*neutral_500",
        input_border_color_dark="*neutral_500",
        input_border_color_focus="*primary_700",
        input_border_color_focus_dark="*primary_300",
        link_text_color="*primary_700",
        link_text_color_dark="*primary_300",
        link_text_color_hover="*primary_800",
        link_text_color_hover_dark="*primary_200",
        button_primary_background_fill="*primary_700",
        button_primary_background_fill_dark="*primary_300",
        button_primary_background_fill_hover="*primary_800",
        button_primary_background_fill_hover_dark="*primary_200",
        button_primary_text_color="white",
        button_primary_text_color_dark="*neutral_950",
        button_primary_text_color_hover_dark="*neutral_950",
        button_primary_border_color="*primary_700",
        button_primary_border_color_dark="*primary_300",
        button_medium_padding="0.65rem 1.15rem",
        button_medium_text_weight="600",
        block_shadow="none",
        block_shadow_dark="none",
    )


def index_documents(
    files: list[str] | None, chunk_size: int, overlap: int, strategy: str = "Character-based"
) -> tuple[RetrievalPipeline | None, str, str]:
    pipeline = RetrievalPipeline()
    try:
        count = pipeline.index_documents(files or [], chunk_size, overlap, strategy)
    except (ValueError, OSError, RuntimeError) as exc:
        return None, f"Indexing failed: {exc} No collection is indexed.", ""
    return pipeline, f"Indexed {pipeline.document_count} documents and {count} chunks.", ""


def render_results(results: list[SearchResult]) -> str:
    if not results:
        return "<p>No passages found.</p>"
    passages = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        page = f" · Page {chunk.page}" if chunk.page is not None else ""
        diagnostics = chunk.diagnostics
        if diagnostics is None:
            tokens = '<p class="passage-tokens">Token diagnostics unavailable.</p>'
        elif diagnostics.truncated:
            tokens = (
                f'<p class="truncation-warning">{diagnostics.original_tokens} original · '
                f"{diagnostics.processed_tokens} processed · Truncated "
                f"(limit {diagnostics.token_limit} tokens)</p>"
            )
        else:
            tokens = (
                '<p class="passage-tokens"><span aria-hidden="true">'
                f"{diagnostics.original_tokens} / {diagnostics.token_limit} tokens · "
                'No truncation</span><span class="token-description">'
                f"{diagnostics.original_tokens} original tokens, "
                f"{diagnostics.processed_tokens} processed tokens, "
                f"limit {diagnostics.token_limit}; no truncation</span></p>"
            )
        passages.append(
            f'<article class="passage-card"><h3>{rank}. {escape(chunk.filename)}{page}</h3>'
            f'<p class="passage-meta">Chunk <code>{escape(chunk.identifier)}</code> · '
            f'Cosine similarity: <span class="score">{result.score:.4f}</span></p>'
            f'{tokens}<div class="passage-text">{escape(chunk.text)}</div></article>'
        )
    return '<section aria-label="Retrieved passages">' + "".join(passages) + "</section>"


def search_documents(pipeline: RetrievalPipeline | None, query: str, top_k: int) -> tuple[str, str]:
    return search_with_insights(pipeline, query, top_k)[:2]


def index_with_insights(
    files: list[str] | None,
    chunk_size: int,
    overlap: int,
    strategy: str,
) -> tuple[RetrievalPipeline | None, str, str, str]:
    pipeline, status, results = index_documents(files, chunk_size, overlap, strategy)
    insights = render_index_insights(pipeline.insights) if pipeline and pipeline.insights else ""
    return pipeline, status, results, insights


def search_with_insights(
    pipeline: RetrievalPipeline | None,
    query: str,
    top_k: int,
) -> tuple[str, str, str]:
    if pipeline is None:
        return "Index documents before searching.", "", ""
    index_summary = render_index_insights(pipeline.insights) if pipeline.insights else ""
    try:
        results = pipeline.search(query, top_k)
    except (ValueError, OSError, RuntimeError) as exc:
        return f"Search failed: {exc}", "", index_summary
    return (
        f"Retrieved {len(results)} passages, ordered by cosine similarity.",
        render_results(results),
        render_search_insights(results, pipeline.insights)
        + (render_index_diagnostics(pipeline.insights) if pipeline.insights else ""),
    )


def clear_search(pipeline: RetrievalPipeline | None) -> tuple[str, str, str]:
    """Changing top-k invalidates both views of the previous search."""
    insights = render_index_insights(pipeline.insights) if pipeline and pipeline.insights else ""
    return "Search settings changed. Select Search to update passages.", "", insights


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RagLab", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        collection = gr.State(None, time_to_live=3600)
        with gr.Column(elem_id="rag-lab"):
            gr.HTML(
                '<header class="project-header"><h1>Rag<span class="title-accent">Lab</span>'
                '</h1><a href="https://github.com/norfali/ai-playground/tree/main/rag" '
                'target="_blank" rel="noopener noreferrer" '
                'aria-label="View RagLab on GitHub (opens in a new tab)">'
                'View on GitHub <span aria-hidden="true">↗</span></a></header>'
                '<div class="intro"><p class="tagline">A closer look at retrieval.</p>'
                "<p>Index documents, search for related passages and inspect the evidence. "
                "RagLab retrieves text; it does not generate answers.</p></div>"
            )
            with gr.Column(variant="panel", elem_classes="index-controls"):
                gr.Markdown("## Document indexing", elem_classes="section-heading")
                with gr.Row(equal_height=False):
                    files = gr.File(
                        label="Documents — TXT, Markdown or text-based PDF",
                        file_count="multiple",
                        file_types=[".txt", ".md", ".pdf"],
                        type="filepath",
                        scale=1,
                        min_width=260,
                    )
                    with gr.Column(scale=2, min_width=280):
                        strategy = gr.Radio(
                            STRATEGIES, value=STRATEGIES[0], label="Chunking strategy"
                        )
                        with gr.Row():
                            size = gr.Slider(
                                100,
                                2000,
                                value=800,
                                step=50,
                                precision=0,
                                label="Chunk size (characters)",
                                min_width=160,
                            )
                            overlap = gr.Slider(
                                0,
                                500,
                                value=120,
                                step=10,
                                precision=0,
                                label="Overlap (characters)",
                                min_width=160,
                            )
                        gr.Markdown(
                            "Overlap must be smaller than chunk size. Indexing replaces the "
                            "collection. The embedding model downloads on first use; "
                            "documents are processed locally."
                        )
                with gr.Row(equal_height=False, elem_classes="index-actions"):
                    index = gr.Button("Index documents", variant="primary", scale=0, size="md")
                    index_status = gr.Textbox(
                        label="Indexing status",
                        value="No documents indexed.",
                        interactive=False,
                        container=False,
                        scale=1,
                    )

            gr.Markdown("## Semantic search", elem_classes="section-heading")
            with gr.Row(elem_classes="search-controls", equal_height=False):
                query = gr.Textbox(
                    label="Search query",
                    placeholder="What is semantic similarity?",
                    scale=4,
                    min_width=260,
                )
                top_k = gr.Slider(
                    1,
                    20,
                    value=5,
                    step=1,
                    precision=0,
                    label="Top-k passages",
                    scale=1,
                    min_width=160,
                )
                search = gr.Button("Search", variant="primary", scale=0, size="md")
            search_status = gr.Textbox(label="Search status", interactive=False, container=False)
            with gr.Row(elem_classes="retrieval-workspace", equal_height=True):
                with gr.Column(scale=3, min_width=0, elem_classes="workspace-panel"):
                    gr.Markdown("## Retrieved Passages", elem_classes="section-heading")
                    results = gr.HTML(
                        "<p>Index documents and enter a query to see passages.</p>",
                        elem_id="retrieved-passages",
                        elem_classes="workspace-content",
                        html_template='<div class="panel-scroll" tabindex="0" role="region" '
                        'aria-label="Retrieved passages">${value}</div>',
                    )
                with gr.Column(scale=2, min_width=0, elem_classes="workspace-panel"):
                    gr.Markdown("## Retrieval Insights", elem_classes="section-heading")
                    insights = gr.HTML(
                        "<p>Index documents to see measurements and explanations.</p>",
                        elem_id="retrieval-insights",
                        elem_classes="workspace-content",
                        html_template='<div class="panel-scroll" tabindex="0" role="region" '
                        'aria-label="Retrieval insights">${value}</div>',
                    )
            index.click(
                index_with_insights,
                [files, size, overlap, strategy],
                [collection, index_status, results, insights],
                concurrency_id="retrieval",
            ).then(lambda: "", outputs=search_status)
            for event in (search.click, query.submit):
                event(
                    search_with_insights,
                    [collection, query, top_k],
                    [search_status, results, insights],
                    concurrency_id="retrieval",
                )
            top_k.input(
                clear_search,
                [collection],
                [search_status, results, insights],
                concurrency_id="retrieval",
                trigger_mode="always_last",
                show_progress="hidden",
            )
    return app
