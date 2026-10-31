
# RagLab

**A closer look at retrieval.**

Explore how machines find information in documents. RagLab is a small, local retrieval playground that splits documents into chunks, generates embeddings and searches for passages by meaning. Experiment with different settings, inspect the results and understand what happens behind the scenes.

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

The first indexing operation downloads the embedding model from Hugging Face and may take longer. Later runs reuse the cached model.

## Tech stack

- **Python 3.12+** — document processing, chunking and retrieval.
- **[Gradio](https://www.gradio.app/)** — browser-based interface for indexing documents and exploring results.
- **[Sentence Transformers](https://www.sbert.net/)** — local text embeddings using `all-MiniLM-L6-v2`.
- **[NumPy](https://numpy.org/)** — vector operations and cosine similarity search.
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — extracting text from PDF documents.
- **[uv](https://docs.astral.sh/uv/)** — dependency and workspace management.
- **Ruff and pytest** — linting, formatting and automated tests.

## How it works

1. **Upload documents.** Add TXT, Markdown or text-based PDF files. PDF pages are processed separately to preserve source references.
2. **Choose a chunking strategy.** Split documents using character-based or sentence-aware chunking, with configurable size and overlap.
3. **Index documents.** RagLab generates embeddings locally and stores them in an in-memory index. The default model is `all-MiniLM-L6-v2`.
4. **Search by meaning.** Enter a question and retrieve the most semantically similar passages. Top-k defaults to five results.
5. **Explore the results.** Inspect source text, similarity scores, token usage and Retrieval Insights to understand what was found and what to try next.

Each indexing operation replaces the previous collection. Document embeddings are reused across searches, while the query receives its own embedding.

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
| **RAG** | Retrieval-Augmented Generation: finding information to support an AI-generated answer. RagLab currently performs retrieval only. |
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

Document processing, embeddings, retrieval and insights are kept separate from the UI. Tests use fake embeddings and generated PDFs; they do not download model weights.

## Privacy

Document processing and embeddings run locally. RagLab does not send uploaded documents to a cloud inference service. The CLI binds to the local machine, disables public sharing and turns off Gradio analytics and Hugging Face telemetry.

The embedding model is downloaded from Hugging Face when first needed and cached locally. Each browser session has its own in-memory document index, which is lost when the application restarts. Embeddings and indexes are not persisted to disk.

Uploaded documents are stored temporarily by Gradio and may remain after use until cleanup. Take care with sensitive documents, especially on shared devices.

## Limitations

RagLab is designed for small local document collections. It does not support OCR, encrypted PDFs, complex document layout reconstruction or persistent indexes. PDF extraction quality depends on the document's reading order.

Semantic search always returns the closest available passages, even when none genuinely answers the question. Similarity scores are not measures of factual correctness, and Retrieval Insights cannot independently determine relevance or answer quality.

RagLab currently performs **retrieval only**. It does not generate answers, verify claims or calculate retrieval evaluation metrics such as precision and recall.

## Disclaimer

RagLab is an educational exploration, provided “as is” without warranty. Retrieved passages and similarity scores may be incomplete, misleading or irrelevant; independently verify source information before using it for consequential decisions.

You are responsible for the rights and permissions needed to process uploaded documents and for complying with applicable privacy laws. To the extent permitted by law, the author accepts no liability for loss or damage arising from use.

## License

Released under the [MIT License](../LICENSE).
