
# RagLab

**A closer look at retrieval.**

Explore how machines find information in documents and use it to answer questions. RagLab splits documents into chunks, generates embeddings and searches for passages by meaning. Optional local answer generation makes the complete RAG pipeline inspectable, from source passages to prompt context and stage timings.

[AI Playground](https://github.com/norfali/ai-playground) · [Fenka](https://www.fenka.co.uk)

## Getting started

You'll need:

- Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).
- A Mac, Linux or Windows computer capable of running the embedding model on CPU.

Clone the repository, install the dependencies and launch RagLab:

```bash
git clone https://github.com/norfali/ai-playground.git
cd ai-playground
make sync
make rag
```

Open the local address printed in your terminal, usually [http://localhost:7860](http://localhost:7860). Press `Ctrl+C` to stop the app.

After restarting RagLab following a code update, refresh the browser page before using it again. An already-open page can retain an earlier event configuration and submit the wrong number of inputs. If the terminal reports “didn't receive enough input values”, close the old tab and reopen the local address printed by the current process. Index documents again: the index belongs to the browser session and is not persisted across application restarts. API clients must also reconnect after interface changes and supply the generation option and model directory alongside the question and top-k.

The first indexing operation downloads the embedding model from Hugging Face and may take longer. Later runs reuse the cached model.

## Tech stack

- **Python 3.12+** — document processing, chunking and retrieval.
- **[Gradio](https://www.gradio.app/)** — browser-based interface for indexing documents and exploring results.
- **[Sentence Transformers](https://www.sbert.net/)** — local text embeddings using `all-MiniLM-L6-v2`.
- **[NumPy](https://numpy.org/)** — vector operations and cosine similarity search.
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — extracting text from PDF documents.
- **[MLX LM](https://github.com/ml-explore/mlx-lm)** (optional) — local answer generation on Apple Silicon, using an existing model directory.
- **[uv](https://docs.astral.sh/uv/)** — dependency and workspace management.
- **Ruff and pytest** — linting, formatting and automated tests.

## How it works

1. **Upload documents.** Add TXT, Markdown or text-based PDF files. PDF pages are processed separately to preserve source references.
2. **Choose a chunking strategy.** Split documents using character-based or sentence-aware chunking, with configurable size and overlap.
3. **Index documents.** RagLab generates embeddings locally and stores them in an in-memory index. The default model is `all-MiniLM-L6-v2`.
4. **Search by meaning.** Enter a question and retrieve the most semantically similar passages. Top-k defaults to five results.
5. **Explore the results.** Inspect source text, similarity scores, token usage and Retrieval Insights to understand what was found and what to try next.
6. **Optionally generate an answer.** Enable local generation and enter a model directory. Inspect the answer, its references, the assembled source context and stage timings.

Each indexing operation replaces the previous collection. Document embeddings are reused across searches, while the query receives its own embedding.

## The two pipelines

```text
Ingestion: Documents → Parsing → Chunking → Embeddings → In-memory index
Query:     Question → Query embedding → Retrieval → Context assembly → Optional local LLM → Answer
```

`IngestionPipeline` in `pipeline.py` composes the existing document loader, chunker, embedding adapter and NumPy index. `RetrievalPipeline` retains the indexing/search interface and exposes query embedding and retrieval separately. `QueryPipeline` composes those query stages, context assembly and optional generation. None of these modules depends on Gradio.

`context.py` assigns source labels in retrieval order: `[S1]`, `[S2]`, and so on. Each label maps to a chunk identifier, filename, PDF page when available and exact character offsets within the document or page. Passage text is preserved verbatim. Empty passages are omitted.

With generation disabled, all nonblank retrieved passages form an **unbudgeted context preview**. This path never imports MLX LM or loads an LLM/tokenizer. It remains usable on the platforms supported by the embedding model.

With generation enabled, `generation.py` loads the configured local model and uses **that model's tokenizer** to count the complete chat prompt, including system instructions, question, source headers and special tokens. RagLab reserves **512 answer tokens** and uses a window of at most **4,096 tokens**, respecting smaller limits declared by the model or tokenizer. Models with no identifiable context limit are rejected.

Whole chunks are considered in rank order. A chunk that would exceed the prompt budget is omitted, and subsequent smaller chunks can still fit. Text is never silently shortened; labels retain their original retrieval rank. No character-to-token estimate is used. This is deliberately simple packing, without reranking or overlap deduplication.

The model is instructed to use only the supplied passages, cite supported claims and return **“Insufficient evidence in the retrieved context.”** when they do not answer the question. Empty context skips generation. Similarity scores alone cannot establish whether the evidence is sufficient, so there is no fixed score cutoff.

## Optional local answer generation

The supported generation runtime is **MLX LM on Apple Silicon macOS**. It runs directly inside RagLab, with no model server, cloud API or API key. Quantized models reduce weight memory, but generation also needs memory for its context and working buffers. Smaller models are easier to run but less reliable at following instructions and citing evidence.

From the repository root, install the optional Python dependencies:

```bash
uv sync --all-packages --extra llm
make rag
```

This installs runtime packages, **not model weights**. A later `make sync` omits optional extras; rerun the command above if necessary. You can also launch with explicit extras:

```bash
uv run --package rag-lab --extra llm rag-lab
```

A lightweight starting model is [`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit), an Apache-2.0 instruction model converted to MLX format. It is a learning baseline, not a guarantee of answer quality.

Download its files yourself, only when you want generation. For example, the Hugging Face CLI supplied by the dependencies can download to a directory outside the repository:

```bash
uv run --package rag-lab hf download mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --local-dir /absolute/path/outside/repository/qwen2.5-1.5b-instruct-4bit
```

That command explicitly downloads weights and requires network access. Skip it if you already have a complete compatible model directory. Keep the configuration, tokenizer files, chat template and safetensors weights together.

In RagLab:

1. Index documents and enter a question.
2. Open **Local model settings** and enter the existing model directory.
3. Enable **Generate answer with local LLM**, then select **Search** or press Enter in the question field.

RagLab accepts local directories, not Hub repository IDs. The tokenizer loads local files with remote tokenizer code disabled. LLM loading is lazy, and the most recently loaded runtime is cached in process. Inference is serialized and each query starts with a fresh prompt; there is no conversational memory. Restart RagLab to release the cached model or reload weights changed at the same path.

An absent optional package, missing/incomplete model, unsupported model or generation error produces an actionable status while retaining the retrieved passages and context. No LLM weights are automatically downloaded, including on failure. Existing embedding-model first-use downloads are unchanged.

## Inspecting a query

The original **Retrieved Passages** and **Retrieval Insights** panels remain available. **Generated answer** displays plain text. **Answer sources** distinguishes references cited by the model from other passages supplied in context. Unknown citation labels and answers with no valid references produce visible warnings. Citation checks establish label membership only; they do not verify factual support.

Open **Pipeline inspection — context and diagnostics** to inspect the exact assembled source text and local JSON metadata:

- Query embedding, vector retrieval, context assembly, model loading, generation and total durations in seconds. `null` means the stage was skipped.
- Retrieved chunk identifiers and cosine scores, included/omitted chunk identifiers, and the configured model directory when generation was requested.
- Full prompt token count, context window and reserved answer budget when prompt preparation succeeds.
- Query status and citation warnings.

Retrieval timing excludes query embedding, which has its own measurement. Context assembly includes the initial preview and tokenizer-based packing when enabled. Model loading timing includes cache lookup or a first load; generation timing covers the generation call. Total time includes orchestration overhead. Failed generation/loading attempts still have timings. These measurements use a monotonic clock and are not hardware benchmarks.

Editing the question, top-k, generation option or model directory clears previous query outputs. Indexing also clears answers, sources and diagnostics. Diagnostics stay in the local session/interface; no traces are sent to an external service or written to a tracing database.

## Exploring chunking

Chunking divides a document into smaller passages before generating embeddings. The strategy, size and overlap affect what information can be retrieved together.

- **Character-based** splits text into windows of up to the configured size, preferring whitespace near boundaries. It uses a fixed character overlap but can split sentences.
- **Sentence-aware** tries to preserve complete sentences, headings and list items. Its overlap is a maximum budget rather than a guarantee, so adjacent chunks may share less text than requested.

The defaults are **800 characters with 120 characters of overlap**. Chunk sizes are measured in characters, not model tokens.

Smaller chunks can retrieve more focused evidence but may separate useful context. Larger chunks can preserve more context while combining unrelated topics or exceeding the embedding model's token limit. Neither strategy guarantees better retrieval.

Try indexing the same document with different settings:

| Strategy | Size | Overlap | What to explore |
| --- | ---: | ---: | --- |
| Character-based | 800 | 120 | Baseline passages and sentence boundaries |
| Sentence-aware | 800 | 120 | More natural boundaries and context |
| Sentence-aware | 2,000 | 120 | Longer passages and possible token truncation |

Repeat the same search after each indexing operation and compare the retrieved passages. RagLab does not yet save or compare experiments automatically.

## Retrieval Insights

Retrieval Insights explains what happened during indexing and searching using measurements and deterministic rules, without an LLM.

The interface displays retrieved passages and insights side by side, with independently scrollable panels on desktop.

### What RagLab measures

- **Top similarity:** the highest cosine similarity score among the retrieved passages.
- **Score margin:** the difference between the first two scores, showing how closely they are ranked.
- **Token truncation:** how many indexed chunks exceed the embedding model's effective token limit.
- **Chunk statistics:** document and chunk counts, lengths, configuration and source overlap.

The default embedding model processes up to **256 tokens** per input, including special tokens. RagLab uses the model's actual tokenizer to detect chunks that exceed this limit. Text beyond the limit may appear in a retrieved passage without contributing to its embedding.

### Understanding the results

Insights are organised into three questions:

- **What happened?** Summarises the search and its measurements.
- **What does it mean?** Explains the observations and their limitations.
- **What should I try next?** Suggests a practical experiment, such as changing chunk size or strategy.

A small difference between similarity scores doesn't mean the passages are equally relevant. Likewise, a high score doesn't prove that a passage answers the question.

RagLab uses heuristics to suggest experiments, not to judge factual correctness. It cannot explain exactly why an embedding model produced a particular score.

## Good to know

| Term | Meaning |
| --- | --- |
| **RAG** | Retrieval-Augmented Generation: retrieving source passages and supplying them to a model to support an answer. Generation is optional in RagLab. |
| **Chunk** | A smaller passage of text extracted from a document. |
| **Chunk size** | The maximum length of a passage, measured in characters in RagLab. |
| **Overlap** | Text repeated between adjacent chunks to preserve context. |
| **Embedding** | A numerical representation of text that captures aspects of its meaning. |
| **Semantic search** | Finding passages by similarity of meaning rather than exact keywords alone. |
| **Cosine similarity** | Measures how closely two embedding vectors align. A score of 0.4138 is not 41.38% confidence. |
| **Top-k** | The maximum number of passages returned by a search. |
| **Token** | A unit of text processed by a model; not necessarily a whole word. |
| **Truncation** | When text exceeds the model's input limit and some content is excluded from its embedding. |

## Development

From the repository root:

```bash
make sync       # Install workspace dependencies
make rag        # Launch RagLab
make rag-test   # Run RagLab tests
make lint       # Check Python linting and formatting
make format     # Format Python code
```

Document processing, embeddings, retrieval, context assembly, generation and insights are kept separate from the UI. Tests use fake embeddings, tokenizers, runtime modules and LLM responses, along with generated PDFs. They do not download weights or run live inference. The root `uv.lock` is the workspace dependency lockfile; MLX LM is an optional, Apple Silicon-specific extra.

## Privacy

Document processing, embeddings and optional answer generation run locally. RagLab does not send uploaded documents or questions to a cloud inference service. The CLI binds to the local machine, disables public sharing and turns off Gradio analytics and Hugging Face telemetry.

The embedding model is downloaded from Hugging Face when first needed and cached locally. Each browser session has its own in-memory document index, which is lost when the application restarts. Embeddings and indexes are not persisted to disk.

Uploaded documents are stored temporarily by Gradio and may remain after use until cleanup. Take care with sensitive documents, especially on shared devices.

## Limitations

RagLab is designed for small local document collections. It does not support OCR, encrypted PDFs, complex document layout reconstruction or persistent indexes. PDF extraction quality depends on the document's reading order.

Semantic search always returns the closest available passages, even when none genuinely answers the question. Similarity scores are not measures of factual correctness, and Retrieval Insights cannot independently determine relevance or answer quality.

Generated answers can hallucinate, follow malicious instructions embedded in source text or cite passages that do not support their claims. Grounding instructions and citation-label checks reduce ambiguity but cannot verify an answer. Unsupported or uncited outputs are not automatically rewritten. Small models may fail to abstain even when evidence is insufficient.

Generation currently supports local MLX instruction models with a chat template and a declared context limit. The 4,096-token application cap and whole-chunk packing can omit useful evidence, including a top-ranked chunk that cannot fit. Generation is non-streaming and uses greedy decoding, which does not guarantee identical results across runtime/hardware versions. Real inference quality and performance require a manual check on Apple Silicon with your chosen weights; automated tests mock inference.

There is no claim verification, retrieval evaluation, reranking, persistent query history or external tracing. Possible later experiments include retrieval benchmarks, reranking, streaming answers and explicit workflow comparisons, without adding them to this phase.

## Disclaimer

RagLab is an educational exploration, provided “as is” without warranty. Retrieved passages and similarity scores may be incomplete, misleading or irrelevant; independently verify source information before using it for consequential decisions.

You are responsible for the rights and permissions needed to process uploaded documents and for complying with applicable privacy laws. To the extent permitted by law, the author accepts no liability for loss or damage arising from use.

## License

Released under the [MIT License](../LICENSE).
