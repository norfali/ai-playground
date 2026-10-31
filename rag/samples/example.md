# Retrieval fundamentals

## Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a search step with a language model.
The retrieval step finds relevant passages in a document collection. A later
generation step can use those passages as evidence when composing an answer.
RagLab Phase 1 explores retrieval alone: its results are original passages,
not generated answers.

## Document chunking

Chunking splits documents into smaller passages so that each embedding describes
a focused piece of content. Small chunks can lose context; large chunks can mix
unrelated ideas. Overlap repeats some text between adjacent chunks to preserve
context at boundaries. RagLab measures chunk size and overlap in characters and
keeps PDF pages separate so a passage can be traced to a page.

## Embeddings

An embedding model maps text to a vector of numbers. Related meanings tend to
produce nearby vectors, even when the texts use different words. RagLab uses
all-MiniLM-L6-v2 locally to embed both document chunks and search queries. A model
has a limited token window, so long passages may be truncated during embedding.
The complete chunk text is still displayed in retrieval results.

## Semantic similarity

Cosine similarity measures the angle between vectors. With unit-length vectors,
cosine similarity is their dot product. Larger scores indicate closer directions;
they are not probabilities or calibrated confidence scores. Similarity does not
prove that a passage is correct, relevant or sufficient to answer a question.

## Vector search

Vector search compares a query embedding with stored document embeddings and
returns the top-k passages in descending similarity order. RagLab uses an exact
NumPy scan over an in-memory matrix, which is simple to inspect for small
collections. Document vectors are computed once per indexing operation, while
each search only needs a new query embedding. Re-indexing replaces the collection.

## Retrieval evaluation

Retrieval evaluation needs queries and judgments about which passages are relevant.
Recall@k measures how many relevant passages appear among the first k results.
Mean Reciprocal Rank (MRR) averages the reciprocal rank of the first relevant
result for each query. These measurements can compare chunking strategies,
embedding models and ranking methods. Phase 1 displays similarity scores but does
not yet implement an evaluation benchmark.
