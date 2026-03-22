# FDA Label Retrieval Design: Corpus, Indexing, and Baseline Comparison

## Summary
Design Milestone 6 around a local FDA drug-label corpus with two retrieval paths over the same normalized chunks:

- `grep`-style free-text scan as the naive baseline
- BM25 over an indexed local corpus as the primary retrieval method

The assistant remains a **single-drug label Q&A** system in v1. Retrieval happens only after resolving one drug name to one label record.

## Corpus and Ingestion
- Use FDA/openFDA drug-label data as the source corpus, but ingest from the **official downloadable dataset / repository artifacts**, not by scraping the entire API.
- Normalize each label into:
  - `label_id`
  - `drug_name_primary`
  - `drug_name_aliases`
  - `product_type`
  - `effective_date`
  - `section_name`
  - `section_text`
- Preserve raw source provenance so each answer can cite the underlying label and section.
- Treat each label as a living document version; v1 should use the **latest available label version per resolved drug**, not historical version comparison.

## Chunking Schema
- Primary chunking policy: **section-aware chunking**.
- Each canonical section becomes at least one chunk:
  - `boxed warning`
  - `indications and usage`
  - `dosage and administration`
  - `contraindications`
  - `warnings and precautions`
  - `adverse reactions`
  - `drug interactions`
  - `use in specific populations`
- If a section is long, split it into overlapping subchunks while preserving:
  - `chunk_id`
  - `label_id`
  - `section_name`
  - `chunk_order`
  - `chunk_text`
- Do not use blind fixed windows across the whole document unless a section cannot be identified.

## Retrieval Architecture
- Step 1: resolve the user’s `drug_name` to one label.
- Step 2: restrict all retrieval to chunks from that label only.
- Step 3: run one of two retrieval methods on those chunks.
- Step 4: return top `k` chunks with section names and scores.
- Step 5: synthesize a short narrative answer citing the sections used.

### Baseline 1: Free-Text Search
- Implement a simple literal/regex text scan over the local chunks.
- Ranking policy for the baseline:
  - exact phrase hits first
  - then hit count / earliest occurrence / section-priority tie-breakers
- This is intentionally naive and mostly for comparison and pedagogy.
- The baseline should operate on **chunks**, not whole files, so the comparison with BM25 is fair.

### Baseline 2: Indexed BM25
- Build a lexical index over chunk text.
- Include optional field boosts for:
  - `section_name`
  - `drug_name_primary`
- Use BM25 scores to rank the top `k` chunks within the resolved label.
- Preferred local implementation:
  - SQLite + FTS5 if available, or
  - a small explicit BM25 library over normalized chunk text
- The implementation should expose scores so retrieval quality is inspectable.

## Tooling Contract
- Primary tool: `query_label(drug_name, question, top_k=5)`
- Tool behavior:
  - resolve the drug name
  - retrieve top chunks from that label with the selected method
  - return:
    - resolved drug name
    - label ID
    - retrieval method
    - top matches with section names, scores, and text
- The system prompt should instruct the model to:
  - always use the tool for label-content questions
  - avoid answering from memory
  - ask for a drug name if missing
  - reject multi-drug or cross-label requests in v1

## Evaluation and Comparison
- Build a small eval set of 15-20 intuitive queries across:
  - side effects
  - contraindications
  - pregnancy
  - interactions
  - warnings
  - dosing
- For each query, compare:
  - whether `grep` returns the correct section in the top results
  - whether BM25 ranks the most relevant section higher
  - whether the final narrative answer is grounded in the right section
- Include failure-focused cases where lexical mismatch is modest but still human-interpretable, such as:
  - “side effects” vs `adverse reactions`
  - “who should not take it” vs `contraindications`
  - “pregnancy warning” vs `use in specific populations`

## Important Decisions
- Full-corpus ingestion should use official downloadable data, not API pagination.
- The API may still be used later for incremental updates or metadata checks.
- v1 retrieval is lexical only; no vector database or embeddings.
- `grep` is a teaching baseline, not the primary retrieval engine.
- The product is a label-grounded information assistant, not a medical advice system.

## Acceptance Criteria
- A local corpus can be built without depending on bulk API crawling.
- The same chunk set supports both free-text scan and BM25 retrieval.
- Single-drug queries can be answered with cited sections.
- The retrieval comparison is measurable on a fixed eval set and demonstrates why indexing/ranking is better than naive scan.

## Assumptions
- FDA/openFDA label records are large enough that indexing is useful, but still small enough for local experimentation on a laptop.
- The downloadable repository artifacts remain available and are the preferred lawful/practical source for full ingestion.
- Historical version handling is deferred; v1 uses the latest available label per drug.
## Milestone 6: Retrieval and Knowledge Access

This milestone teaches the agent to retrieve grounded knowledge from an
external corpus instead of pretending that prompt context is enough.

For this repo, the first retrieval corpus is FDA drug labels. This keeps the
questions intuitive while still showing the mechanics of chunking, ranking,
citations, and retrieval evaluation.

### Use Case

The v1 retrieval assistant is a single-drug FDA label Q&A system.

Example questions:

- What are the common adverse reactions for sertraline?
- What does the label say about pregnancy for ibuprofen?
- What contraindications are listed for warfarin?
- What drug interactions are mentioned for sertraline?

The agent should answer those questions by retrieving relevant sections from
the named drug label and synthesizing a short grounded response.

### Corpus And Ingestion

The source corpus is FDA/openFDA drug labeling data, but the design assumes
that full local ingestion comes from official downloadable repository artifacts
instead of crawling the API.

Each normalized label record keeps:

- `label_id`
- `drug_name_primary`
- `drug_name_aliases`
- `product_type`
- `effective_date`
- `source_path`

For v1, the local corpus keeps only the latest available label version for each
resolved primary drug name.

### Chunking

Chunking is section-aware rather than based on blind fixed token windows.

Preferred canonical sections:

- boxed warning
- indications and usage
- dosage and administration
- contraindications
- warnings and precautions
- adverse reactions
- drug interactions
- use in specific populations

Each section becomes at least one chunk. Very long sections may be split into
overlapping subchunks, but the section name is always preserved for citation
and ranking.

### Retrieval Design

Retrieval is deliberately narrow in v1:

1. resolve the user-supplied drug name to one label
2. search only inside that label
3. return top `k` matching chunks
4. answer from those chunks with citations

The repo implements two retrieval backends over the same normalized chunks:

- `grep`-style free-text scan as the naive baseline
- BM25 ranking as the primary indexed retrieval method

This comparison is useful for learning because `grep` shows how far a direct
text scan can get, while BM25 shows why indexing and ranking matter even for a
local corpus that still fits on a laptop.

### Tool Contract

The primary retrieval tool is `query_label(drug_name, question, top_k=5)`.

The tool:

- resolves one brand or generic drug name
- retrieves the best matching chunks from that label
- returns section names, scores, and chunk text

The system prompt should tell the model to:

- use `query_label` for label-content questions
- avoid answering from memory
- ask for a drug name if missing
- reject multi-drug or broad cross-label queries in v1

### Scope Boundaries

This milestone is intentionally constrained:

- one named drug per query
- no cross-drug comparison
- no whole-corpus search as the main product surface
- no embeddings or vector database
- no medical advice beyond reporting what the label says

### Evaluation

Evaluation should stay small and inspectable.

Maintain a local set of intuitive questions covering:

- side effects
- contraindications
- pregnancy
- interactions
- warnings
- dosing

For each question, compare:

- whether `grep` returns the expected section in the top results
- whether BM25 ranks the expected section higher
- whether the final answer is grounded in the right supporting text

Representative mismatch cases are especially useful:

- side effects versus `adverse reactions`
- who should not take it versus `contraindications`
- pregnancy warning versus `use in specific populations`

### What This Teaches

- the difference between lookup and retrieval
- why chunking matters
- why ranking matters
- how citations improve debugging
- why a simple lexical baseline is often the right first step before vectors
