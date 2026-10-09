# Intelligent Document Retrieval & Evaluation Platform

Milestones 1–5 provide a FastAPI service for document ingestion, semantic and BM25 retrieval, hybrid rank fusion, and a reproducible retrieval evaluation harness.

## Requirements and setup

- Python 3.10 or newer
- Windows PowerShell
- First semantic search use needs internet access to download the embedding model. Hugging Face caches the weights for later runs.

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"
    python -m uvicorn app.main:app --reload

The API is at http://127.0.0.1:8000 and interactive docs are at /docs. SQLite defaults to data/documents.db. Override DATABASE_PATH, MAX_UPLOAD_BYTES, CHUNK_SIZE, or CHUNK_OVERLAP through environment variables.

## Tests

    python -m pytest

Tests for semantic retrieval use the actual pretrained model and may download its weights on first use.

## API examples

Upload with curl.exe -X POST "http://127.0.0.1:8000/documents" -F "file=@sample.txt". The response contains id, name, file_type, size_bytes, status, created_at, and chunk_count. GET /documents returns {"documents": [...]}; GET /documents/{id} returns document metadata and chunks. GET /health returns {"status":"ok"}.

Semantic search remains at GET /search?q=...&top_k=5. For lexical search, run:

    curl.exe --get "http://127.0.0.1:8000/search/lexical" --data-urlencode "q=AUTH-9321" --data-urlencode "top_k=5"

The lexical response contains query, top_k, and a results list. Each result includes document_name, chunk_id, text, page_number, section, and the computed bm25_score. Either search route requires non-whitespace query text and accepts top_k from 1 through 100.

## BM25 lexical retrieval

BM25 (Best Matching 25) assigns a relevance score using the query terms that also occur in each chunk. Term frequency increases a chunk's score, but with diminishing returns, so repeating a word many times does not increase the score without limit. Inverse document frequency (IDF) gives more weight to rarer terms, since a rare identifier is usually more informative than a common word. BM25 also adjusts for chunk length so long chunks do not win just because they contain more words.

This makes BM25 useful for exact identifiers, product names, error codes, and technical vocabulary. The tokenizer lowercases words while keeping common identifier separators such as hyphens, underscores, dots, and slashes. BM25 is lexical: it needs terms to overlap. Semantic search compares embedding directions and can match related wording without exact token overlap. The evaluation section below compares these two baselines with hybrid retrieval on the checked-in dataset.

The project uses the small rank-bm25 package's BM25Okapi implementation. Token lists are persisted as JSON in SQLite in lexical_chunks, keyed by chunk ID and tagged with a tokenizer version. A foreign key with cascade deletion keeps each lexical row tied to the source chunk. New uploads save their lexical tokens in the same SQLite transaction as their chunks. When the lexical endpoint encounters older chunks without tokens, it tokenizes and stores them automatically, so prior documents do not need to be uploaded again.

For simplicity, each lexical search rebuilds a BM25Okapi scorer from the stored token lists. This recomputes corpus statistics on every request, which is easy to inspect and suitable for a small local corpus; a large corpus would benefit from a cached or dedicated inverted index. BM25 scores are relative ranking scores, not probabilities or confidence values.

## Hybrid retrieval with Reciprocal Rank Fusion

GET /search/hybrid?q=...&top_k=5 asks semantic and lexical search for their top-K ranked candidates, then combines those rankings with Reciprocal Rank Fusion (RRF). Use it with:

    curl.exe --get "http://127.0.0.1:8000/search/hybrid" --data-urlencode "q=AUTH-9321" --data-urlencode "top_k=5"

The response includes query, top_k, rrf_constant, and results. Each result includes document_name, chunk_id, text, page_number, section, hybrid_score, semantic_rank, and lexical_rank. A missing rank means that retriever did not return the chunk. Candidates appearing in both lists are counted once and receive contributions from both ranks.

RRF uses ranks rather than raw BM25 or cosine scores, since those score scales are not directly comparable:

    RRF_score(document) = sum over retrievers r of 1 / (k + rank_r(document))

The configured constant is k=60 and rank positions begin at 1. For example, a chunk ranked 1st by semantic and 4th by lexical receives 1/61 + 1/64. Another chunk ranked 2nd by both receives 1/62 + 1/62, which is slightly higher and therefore ranks first. Missing ranks contribute zero. Ties sort by best component rank and then chunk ID, making ordering repeatable.

RRF helps combine complementary evidence: semantic search can find related wording, while BM25 can strongly match exact identifiers. It does not prove the combined method improves retrieval; that requires a later evaluation milestone. The endpoint requests top_k candidates from each method, so its pool is at most 2 × top_k chunks before fusion.
## Semantic search concepts

An embedding is a list of numbers that represents the meaning of a piece of text. The all-MiniLM-L6-v2 model creates 384-number vectors for chunks and questions. Cosine similarity compares vector direction; after normalization, a dot product gives that score. Semantic search calculates a score for every stored chunk and returns the top K. Its exact scan is simple for a small corpus, while its work grows linearly with the number of chunks.

Vectors are stored in SQLite as float32 BLOBs, mapped by chunk ID and tagged with the model name. New uploads are embedded before the document/chunks/vectors are committed together. Semantic search automatically indexes older chunks without vectors. The model loads lazily and is cached once per process; CPU inference avoids requiring CUDA. Sentence Transformers and PyTorch are sizeable dependencies, and model weights require a one-time network download.

## Module guide

- app/config.py holds environment-backed ingestion settings. Interview concepts: immutable configuration and environment-based deployment.
- app/parsing.py validates file structure and extracts text using pypdf, python-docx, or UTF-8 decoding. normalize_text preserves paragraph breaks; extract_text retains PDF pages. Interview concepts: parsing boundaries and input validation.
- app/chunking.py splits text near paragraph boundaries and applies overlap. create_chunks assigns IDs and carries source metadata. Interview concepts: chunk-size tradeoffs and metadata propagation.
- app/embeddings.py loads all-MiniLM-L6-v2 lazily and caches one model instance per process. Interview concepts: inference, caching, and runtime tradeoffs.
- app/semantic_search.py prepares normalized embeddings, catches up missing vectors, and calculates top-K cosine scores. Interview concepts: embeddings and vector similarity.
- app/lexical_search.py tokenizes chunks, catches up missing term rows, and scores the corpus with BM25Okapi. tokenize preserves selected identifier punctuation; prepare_terms creates index data; search ranks matching chunks. Interview concepts: term frequency, IDF, length normalization, and lexical versus semantic search.
- app/evaluation/metrics.py calculates Recall@K, Precision@K, and MRR@K with shared duplicate and unanswerable conventions. app/evaluation/dataset.py validates labels against corpus files; app/evaluation/runner.py builds the corpus and runs the experiment; app/evaluation/experiment.py evaluates cases and writes deterministic JSON/CSV. Interview concepts: metric denominators, macro averages, and evaluation bias.
- app/hybrid_search.py contains fuse_rankings, which adds reciprocal contributions from one-based ranks, deduplicates by chunk ID, and sorts ties deterministically. HybridSearchService calls both existing search services. Interview concepts: rank fusion and why raw score scales should not be mixed.
- app/storage.py owns the SQLite schema and transactions. It stores chunks alongside semantic vectors and lexical terms, each tied to chunk IDs. Interview concepts: foreign keys, BLOB/JSON storage, transactions, and index consistency.
- app/main.py defines health, upload, list, detail, semantic search, and lexical search routes. A shared search_parameters dependency applies the same validation to both search modes. Interview concepts: API validation and separation of route, service, and storage responsibilities.
- app/models.py defines parsed-page and chunk records shared between modules. Interview concepts: dataclasses and type hints.

## Retrieval evaluation

The checked-in evaluation set contains 30 manually authored questions and eight small TXT corpus documents. Each case has a category, a short relevance explanation, and relevant_sources that name corpus file stems. The runner sends the files through the production TXT parser and chunker, then resolves each labeled source to the chunk IDs actually generated. The current source documents each produce one chunk. Four cases are explicitly unanswerable and have an empty relevant_sources list.

This is a small, curated development dataset rather than a representative benchmark. Its questions were written from the same eight short documents they label, so the results are useful for checking pipeline behavior but are vulnerable to author and corpus bias. Relevance means “this chunk contains evidence relevant to the question”; it does not measure whether a generated answer is correct. No answer generation is part of this milestone.

The metrics use the same rules for all methods. Recall@K is the number of distinct relevant chunk IDs in the first K divided by the total relevant IDs. Precision@K is distinct relevant results in the first K divided by K; a missing result slot counts as incorrect. MRR@K is the reciprocal of the rank of the first relevant result within K, or zero if none is found. Duplicate retrieved IDs are collapsed in first-seen order before scoring.

For an unanswerable case, the relevant set is empty, and Recall, Precision, and MRR are explicitly set to zero. Returned chunks are never treated as relevant by default. Reports include both all-case macro averages (including those zero scores) and answerable-only averages so the four no-evidence cases remain visible without obscuring answerable-query ranking quality.

Run the experiment from the project root after installing dependencies:

    python -m app.evaluation.runner

Optional candidate depth, K values, and output directory:

    python -m app.evaluation.runner --candidate-depth 10 --ks 1 3 5 --output-dir evaluation/results

The runner uses one isolated temporary SQLite database, the same corpus, and the same candidate depth for semantic, BM25, and hybrid search. The configured depth is 10, larger than every evaluation K; this corpus has only eight chunks, so all three methods can return the full corpus. The report records corpus and dataset SHA-256 hashes, generated chunk IDs, model name and revision, package versions, chunking settings, candidate depth, K values, and RRF constant.

The output files are evaluation/results/evaluation.json and evaluation/results/per_question.csv. JSON stores aggregate metrics and each case's labels, retrieved IDs, scores, and per-K metrics by method. CSV has one row per question and method with the same traceable IDs and metric columns. Re-running with the same inputs and model revision writes deterministic artifacts.

### Measured results for the checked-in corpus

These are actual results from the runner on Python 3.13.0, sentence-transformers/all-MiniLM-L6-v2 (the report does not record a model revision), sentence-transformers 5.7.0, rank-bm25 0.2.2, eight corpus chunks, candidate depth 10, and K values 1, 3, and 5. Values below are answerable-only macro averages over 26 questions; the four unanswerable cases are scored as zero and reported separately in the JSON.

| Method | K | Recall@K | Precision@K | MRR@K |
|---|---:|---:|---:|---:|
| Semantic | 1 | 1.0000 | 1.0000 | 1.0000 |
| BM25 | 1 | 0.9615 | 0.9615 | 0.9615 |
| Hybrid | 1 | 1.0000 | 1.0000 | 1.0000 |
| Semantic | 3 | 1.0000 | 0.3333 | 1.0000 |
| BM25 | 3 | 1.0000 | 0.3333 | 0.9808 |
| Hybrid | 3 | 1.0000 | 0.3333 | 1.0000 |
| Semantic | 5 | 1.0000 | 0.2000 | 1.0000 |
| BM25 | 5 | 1.0000 | 0.2000 | 0.9808 |
| Hybrid | 5 | 1.0000 | 0.2000 | 1.0000 |

On this set, BM25 placed one relevant passage at rank 2 for the incident acknowledgement question, while semantic and hybrid ranked it first; by K=3 BM25 recovered that passage. Semantic and hybrid tie on the reported macro metrics. This small curated dataset does not establish that hybrid retrieval is generally better. At K=5 all answerable methods have full recall, partly because there are only eight corpus chunks and one labeled relevant source per question.

The experiment runner and metric code live in app/evaluation. Dataset source labels are validated against the corpus files, so an unknown source label fails before indexing. Per-question output retains evidence labels and retrieved chunk IDs to make unexpected scores reviewable.
## Placement interview questions

1. What does Recall@K measure? — The share of labeled relevant chunks found among the first K results.
2. How does MRR differ from Recall? — MRR rewards placing the first relevant result near the top; Recall counts how many relevant chunks were found.
3. What does an empty relevance set mean? — The case has no evidence in the corpus, so all three ranking metrics are explicitly zero; returned chunks are not labeled relevant.
4. Why use the same candidate depth for each retriever? — It gives methods the same opportunity to return candidates before applying the same metric cutoffs.
5. What can bias this experiment? — Questions and labels come from a tiny corpus authored for this project, so wording, topic coverage, and relevance judgments may favor the tested examples.
## Current scope and limitations

Reranking, answer generation, deletion endpoints, and a frontend remain later milestones. The evaluation harness is implemented, with results limited by its small, curated dataset. DOCX/TXT page and section fields stay unset when unavailable, and scanned PDFs need OCR. The next useful step is a measured reranking experiment using this harness. Hybrid scores depend on the selected RRF constant and candidate cutoff; they are ranking signals rather than probabilities. BM25 tokenization is intentionally simple and English-oriented; exact identifiers with the listed separators are retained, but specialized tokenization may be needed for other languages or domain-specific syntax.
