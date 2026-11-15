# RagLab — Engineering Notes

A place to keep track of experiments, results and things I've learned while building RagLab.

## What is RagLab?

A local playground for exploring Retrieval-Augmented Generation (RAG).

The idea is straightforward: search documents for relevant information, pass it to a language model and generate an answer backed by sources.

I'm interested in what happens between those steps — how well retrieval works, whether answers are actually supported by the evidence, and where things can go wrong.

## What's implemented

- Document ingestion (TXT, Markdown, PDF) and configurable chunking.
- Semantic search using `all-MiniLM-L6-v2` and an in-memory NumPy vector index.
- Optional local answer generation using MLX LM.
- Source citations, retrieval diagnostics and stage timings.
- Gradio UI for exploring the pipeline.
- Smoke testing and retrieval evaluation.

Everything runs locally, without cloud APIs.

## 14 November 2026 — First LLM smoke test

**Model:** Qwen3-4B-Instruct-2507-4bit (MLX)

Tested three questions against a small fictional document collection.

| Test | Result |
| --- | --- |
| Fully supported | Correct answer with supporting citation |
| Partially supported | Answered the known part and acknowledged missing information |
| Unsupported | Refused to invent an answer |

**Result: 3/3 passed manual assessment.**

The first model load took around 1.6 seconds. Generation took between 0.7 and 2.1 seconds depending on the answer.

**What I learned**

- Local inference is perfectly practical for a small RAG application.
- The model handled missing information well, including partially answerable questions.
- Semantic search still returns results when none actually answers the question.
- Valid citation labels don't necessarily mean the underlying claims are supported.

This was a basic smoke test, not a measure of general answer accuracy. With only two chunks and both retrieved every time, retrieval wasn't really being challenged.

## 15 November 2026 — Retrieval evaluation

Built a separate evaluation runner to measure retrieval quality without involving the LLM.

**Setup:** 18 fictional documents, 26 questions (23 answerable, 3 unsupported), using `all-MiniLM-L6-v2`.

| Metric | Top 1 | Top 3 | Top 5 |
| --- | ---: | ---: | ---: |
| Hit Rate | 95.7% | 95.7% | 100% |
| Recall | 84.8% | 95.7% | 100% |
| MRR | 95.7% | 95.7% | 96.7% |

**Result:** The correct source ranked first for 22 of 23 answerable questions. All expected source documents appeared within the top five.

One interesting failure:

> Who coordinates project BH-731?

The correct document ranked fourth. Another document mentioning BH-731 ranked first, but it didn't contain the coordinator's name.

This looks like a useful example of semantic search struggling to distinguish between documents sharing an exact identifier.

Five questions also required information from two documents. The first result was relevant, but wasn't enough on its own. That explains why Recall@1 was lower than Hit Rate@1.

**What I learned**

- A high Hit Rate doesn't mean all the required evidence was retrieved.
- Semantic similarity isn't the same as factual relevance.
- Retrieval can look excellent overall while still struggling with specific questions.
- Measuring retrieval separately makes it easier to identify where improvements are needed.

I repeated the evaluation with the same configuration and got identical aggregate results.

These are encouraging results, but the dataset is small, synthetic and scored at document level. Longer documents and more difficult distractors should provide a better challenge.

## What's next?

- Investigate why BH-731 ranked fourth.
- Experiment with smaller chunks and sentence-aware chunking.
- Compare semantic retrieval with BM25 keyword search.
- Explore hybrid retrieval and reranking if the results justify them.
- Eventually evaluate answer grounding and citation accuracy more systematically.

The aim isn't to keep adding features. It's to understand what improves retrieval and answer quality, and what simply adds complexity.