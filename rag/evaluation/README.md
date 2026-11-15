# Fictional retrieval dataset, version 1

`dataset.json` defines an ordered corpus of 18 Markdown documents and 26 questions. All names, identifiers, dates and procedures are fictional. Keep the document order stable: the existing index resolves equal scores in ingestion order.

The 23 supported questions cover exact names, dates, identifiers, operating instructions and comparisons. Five questions (`q06`, `q13`, `q16`, `q21`, `q23`) require two distinct sources, hence at least two chunks. Distractors reuse terminology in unrelated contexts: Lark the vessel and theatre, Orchard telemetry and bakery software, archive records and a board game, backup and vehicle batteries, and physical and network bridges.

Each evidence entry has a stable `document_id` and a verbatim quote. All listed sources are required, not alternatives. Several quotes in the same document still count as one relevant document. Quotes make the annotation reviewable and are validated against source text; they are not used to embed queries, rerank results or score passage coverage. With smaller chunks, retrieving any passage from a relevant document earns a document hit even if that passage omits the quoted fact. Inspect the returned text before interpreting it as answer evidence.

`q24`–`q26` are explicitly unsupported, with reasons. They remain in the report with ranked results but have no positive retrieval metrics or relevance judgements. This harness does not classify unsupported questions or measure abstention.

Defaults (800 characters, 120 overlap) produce one chunk per document. The collection tests competition among sources, not long-document segmentation or embedding truncation. Future datasets could add longer sources and passage-level annotations; neither is measured by this baseline.

Change the dataset version when editing documents, questions or annotations. Preserve question/document IDs when their meaning is unchanged. The report hashes the manifest and source-file hashes, so edits remain detectable even if a version bump is forgotten. The dataset is part of this repository checkout; the evaluation command is intended to run from the repository, not a standalone installed wheel.
