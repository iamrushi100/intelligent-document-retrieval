# Project 1 --- Intelligent Document Retrieval & Evaluation Platform

## 1. Project Purpose

### Core idea

Build a practical document-question-answering system whose main
engineering focus is **retrieval quality**, not simply building another
chatbot or calling a vector database.

The system will allow a user to:

1.  Upload documents.
2.  Parse and structure the content.
3.  Split content into meaningful chunks.
4.  Index the chunks using both lexical and semantic retrieval.
5.  Retrieve relevant evidence for a question.
6.  Rerank the retrieved evidence.
7.  Generate an answer grounded in the retrieved evidence.
8.  Show citations back to the source document/page/section.
9.  Refuse or qualify an answer when sufficient evidence is not
    available.
10. Measure retrieval quality using a small evaluation dataset.

### What this project is NOT

This project is deliberately **not**:

-   A generic "chat with PDF" tutorial.
-   A large enterprise knowledge-management platform.
-   A multi-agent system.
-   A knowledge graph project.
-   A complex RAG framework demonstration.
-   A job/document scraping system.
-   A complicated authentication/permissions system.
-   A project that uses many technologies just to appear sophisticated.

The central engineering problem is:

> **How can we retrieve the right evidence from documents reliably, and
> how can we prove that our retrieval system actually works?**

------------------------------------------------------------------------

# 2. Why This Project Is Worth Building

A basic RAG project is common.

The differentiator here is that the project demonstrates:

-   document ingestion
-   information extraction
-   chunking strategy
-   metadata management
-   lexical retrieval
-   semantic retrieval
-   hybrid retrieval
-   reranking
-   grounded generation
-   source citation
-   retrieval evaluation
-   failure handling
-   testing
-   API/backend engineering
-   deployment readiness

The project should therefore be presented as a **retrieval engineering /
document intelligence project**, rather than simply as a "RAG chatbot."

------------------------------------------------------------------------

# 3. Target Resume Positioning

### Recommended project title

**Intelligent Document Retrieval & Evaluation Platform**

Alternative:

**Document Intelligence & Hybrid Retrieval System**

### Suggested technology line

`Python • FastAPI • Hybrid Search • Vector Retrieval • BM25 • Reranking • LLMs • Evaluation`

### Resume story

The project should eventually support a resume statement similar to:

> Built an intelligent document retrieval platform using hybrid
> lexical + semantic search and reranking, with source-grounded LLM
> responses and an evaluation harness to measure retrieval quality.

A stronger version can be used after real measurements exist:

> Designed and evaluated a hybrid document retrieval pipeline combining
> lexical and semantic search with reranking, improving Recall@K from X%
> to Y% on a custom evaluation dataset.

**Never invent the X/Y numbers. They must come from actual
experiments.**

------------------------------------------------------------------------

# 4. High-Level Architecture

``` text
                    ┌───────────────────┐
                    │   User Documents  │
                    │ PDF / DOCX / TXT  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Document Parser   │
                    │ + Metadata        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Chunking Pipeline │
                    └─────────┬─────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │ Lexical Index   │     │ Vector Index    │
        │ BM25            │     │ Embeddings      │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 └───────────┬───────────┘
                             ▼
                    ┌───────────────────┐
                    │ Hybrid Retrieval  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Reranker          │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Context Selection │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ LLM Generation    │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Answer + Sources      Confidence /
             Page / Section        No-answer
```

------------------------------------------------------------------------

# 5. Core Functional Requirements

## 5.1 Document Upload

Support an intentionally small initial set:

-   PDF
-   DOCX
-   TXT

Requirements:

-   Validate file type.
-   Reject empty/invalid files.
-   Store document metadata.
-   Assign a unique document ID.
-   Preserve filename and source information.
-   Track processing status.

Do not build complicated user accounts or permissions in V1.

------------------------------------------------------------------------

# 6. Document Ingestion Pipeline

The ingestion pipeline should be modular.

``` text
File
 ↓
Parser
 ↓
Normalized text
 ↓
Structure detection
 ↓
Chunking
 ↓
Metadata enrichment
 ↓
Embedding
 ↓
Lexical indexing
 ↓
Vector indexing
```

Each chunk should contain metadata such as:

``` text
document_id
document_name
page_number
section
chunk_id
chunk_text
```

The exact metadata will depend on the source format.

------------------------------------------------------------------------

# 7. Chunking Strategy

Do not blindly use one chunk size and call it done.

Start with a sensible baseline.

For example:

-   token/character-based chunking
-   controlled overlap
-   preservation of paragraph/heading boundaries where practical

The project should later allow comparison of chunking configurations.

Example experiment:

``` text
Configuration A
chunk size = X
overlap = Y

Configuration B
chunk size = X2
overlap = Y2
```

The purpose is not to endlessly optimize parameters.

The purpose is to demonstrate that chunking is an engineering decision
that affects retrieval quality.

------------------------------------------------------------------------

# 8. Retrieval System

This is the heart of the project.

## 8.1 Baseline 1 --- Vector Retrieval

Use semantic embeddings to retrieve relevant chunks.

Purpose:

-   establish a baseline
-   demonstrate semantic similarity
-   provide comparison data

------------------------------------------------------------------------

## 8.2 Baseline 2 --- Lexical Retrieval

Use BM25 or another lightweight lexical retrieval mechanism.

Purpose:

-   handle exact terminology
-   handle names, identifiers, technical terms, numbers, etc.
-   establish a second baseline

------------------------------------------------------------------------

## 8.3 Hybrid Retrieval

Combine:

``` text
Lexical score
+
Semantic score
        ↓
Combined ranking
```

The exact weighting should be configurable.

Example:

``` text
hybrid_score =
    alpha * semantic_score
    +
    (1 - alpha) * lexical_score
```

The project should make the weighting a documented engineering choice
rather than a hidden constant.

------------------------------------------------------------------------

# 9. Reranking

After initial retrieval:

``` text
Top N candidates
       ↓
Reranker
       ↓
Top K final chunks
```

The reranker should improve ordering of relevant evidence.

Important:

Do not add a reranker merely because it is fashionable.

We should measure whether it actually improves retrieval quality.

------------------------------------------------------------------------

# 10. Grounded Answer Generation

The LLM receives only the selected evidence.

The prompt should enforce:

1.  Answer using the supplied evidence.
2.  Do not invent unsupported facts.
3.  Cite the relevant source.
4.  If the evidence is insufficient, explicitly say so.
5.  Keep the answer concise and useful.

Example behavior:

### Evidence available

> "The policy requires password rotation every 90 days."

Question:

> "How often must passwords be rotated?"

Answer:

> Passwords must be rotated every 90 days. \[Source: Security Policy,
> p. 4\]

### Evidence unavailable

Question:

> "What is the company's revenue in 2030?"

If the uploaded documents do not contain this information:

> I couldn't find sufficient evidence in the uploaded documents to
> answer this reliably.

This behavior is an important part of the project.

------------------------------------------------------------------------

# 11. Citations

Answers should provide source information such as:

``` text
Source: annual_report.pdf
Page: 17
Section: Financial Results
```

The citation system should be tied to chunk metadata.

The system should never fabricate page numbers.

If page information is unavailable, it should clearly say so.

------------------------------------------------------------------------

# 12. Evaluation --- The Main Differentiator

This is the most important part of Project 1.

Create a small evaluation dataset.

Initial target:

**30--50 questions**

Each evaluation item should contain:

``` text
question
expected/relevant document
expected page/section when available
expected evidence
```

The dataset does not need to be enormous.

It needs to be carefully constructed.

------------------------------------------------------------------------

# 13. Retrieval Metrics

Start with:

### Recall@K

Did the relevant chunk appear in the top K results?

### Precision@K

How many retrieved chunks were relevant?

### MRR

How early did the first relevant result appear?

These metrics should be calculated for the retrieval pipeline.

------------------------------------------------------------------------

# 14. Required Retrieval Experiments

The project should compare at least:

``` text
Experiment 1
Vector retrieval

Experiment 2
BM25 retrieval

Experiment 3
Hybrid retrieval

Experiment 4
Hybrid + reranking
```

Example final report structure:

  System                     Recall@5             MRR Notes
  ------------------- --------------- --------------- --------------------
  Vector                actual result   actual result Semantic baseline
  BM25                  actual result   actual result Lexical baseline
  Hybrid                actual result   actual result Combined retrieval
  Hybrid + Reranker     actual result   actual result Final system

Use real experimental results.

------------------------------------------------------------------------

# 15. Answer-Level Evaluation

Retrieval metrics alone do not tell the entire story.

For a smaller subset of questions, evaluate:

-   answer correctness
-   groundedness
-   citation correctness
-   unsupported-answer rate

This can initially be partly manual.

Do not build a huge automated evaluation framework unless it becomes
necessary.

The objective is to **understand and demonstrate quality**, not to build
another evaluation product.

------------------------------------------------------------------------

# 16. Failure Handling

The system should gracefully handle:

### Invalid document

Return a clear processing error.

### Empty document

Reject or mark as unusable.

### Unsupported format

Return a clear error.

### No relevant retrieval result

Do not force an answer.

### LLM failure

Return a useful error without corrupting the system state.

### Duplicate document

Handle predictably.

### Very large document

Avoid crashing the API; processing should be controlled.

------------------------------------------------------------------------

# 17. Testing

Tests should cover the important backend behavior.

Minimum areas:

-   document ingestion
-   chunk creation
-   metadata preservation
-   retrieval
-   hybrid ranking
-   citation generation
-   no-answer behavior
-   API validation
-   evaluation metrics

The goal is not 100% test coverage.

The goal is confidence in the important system behavior.

------------------------------------------------------------------------

# 18. Backend/API

Use FastAPI.

Possible endpoints:

``` text
POST /documents
GET  /documents
GET  /documents/{id}
DELETE /documents/{id}

POST /documents/{id}/process

POST /query

GET /search

POST /evaluation/run
GET  /evaluation/results
```

Keep the API simple.

Do not create unnecessary microservices.

------------------------------------------------------------------------

# 19. Frontend

A simple web UI is enough.

Main screens/components:

### Documents

-   Upload document
-   List uploaded documents
-   Processing status

### Query

-   Question input
-   Answer
-   Citations
-   Retrieved evidence

### Evaluation

-   Retrieval metrics
-   Experiment comparison
-   Basic charts/tables

The UI is secondary to the retrieval system.

------------------------------------------------------------------------

# 20. Recommended Technology Direction

Keep the stack manageable.

### Backend

-   Python
-   FastAPI

### Document processing

Use appropriate lightweight Python libraries for:

-   PDF
-   DOCX
-   TXT

### Retrieval

-   BM25 implementation
-   embedding model
-   vector index/database

### LLM

Use an API-based LLM initially.

Keep the LLM provider behind a small abstraction so it can be changed
later.

### Frontend

-   HTML
-   CSS
-   Vanilla JavaScript

A frontend framework is not required.

### Testing

-   pytest

### Containerization

-   Docker

### Version control

-   Git
-   GitHub

Do not add LangChain/LangGraph/vector databases/multiple infrastructure
components unless they solve a real problem in this project.

------------------------------------------------------------------------

# 21. Development Phases

## Phase 1 --- Project Skeleton

Goal:

-   repository
-   Python environment
-   FastAPI
-   configuration
-   basic API
-   Git workflow
-   README

Deliverable:

Running FastAPI application.

------------------------------------------------------------------------

## Phase 2 --- Document Ingestion

Implement:

-   upload
-   parsing
-   normalization
-   metadata
-   chunking
-   persistence

Deliverable:

A document can be converted into searchable chunks.

------------------------------------------------------------------------

## Phase 3 --- Semantic Retrieval

Implement:

-   embeddings
-   vector indexing
-   similarity search
-   top-K retrieval

Deliverable:

A question returns relevant chunks.

------------------------------------------------------------------------

## Phase 4 --- BM25 Retrieval

Implement:

-   lexical indexing
-   BM25 search
-   ranking

Deliverable:

A second independent retrieval baseline.

------------------------------------------------------------------------

## Phase 5 --- Hybrid Retrieval

Implement:

-   semantic + lexical combination
-   configurable weighting
-   ranking

Deliverable:

Hybrid retrieval pipeline.

------------------------------------------------------------------------

## Phase 6 --- Reranking

Implement:

-   candidate retrieval
-   reranking
-   final context selection

Deliverable:

Hybrid + reranker pipeline.

------------------------------------------------------------------------

## Phase 7 --- Grounded Generation

Implement:

-   LLM integration
-   evidence-only prompting
-   citations
-   no-answer behavior

Deliverable:

Reliable document Q&A.

------------------------------------------------------------------------

## Phase 8 --- Evaluation Harness

Implement:

-   evaluation dataset
-   Recall@K
-   Precision@K
-   MRR
-   experiment runner
-   result storage/report

Deliverable:

Actual evidence showing which retrieval approach performs best.

------------------------------------------------------------------------

## Phase 9 --- Testing & Failure Handling

Implement:

-   unit tests
-   integration tests
-   validation
-   failure cases

Deliverable:

Stable backend.

------------------------------------------------------------------------

## Phase 10 --- UI & Deployment

Implement:

-   simple frontend
-   Docker
-   deployment
-   documentation

Deliverable:

Portfolio-ready application.

------------------------------------------------------------------------

# 22. What We Should NOT Build

Explicitly avoid scope creep.

### No:

-   Multi-agent system
-   Agentic workflows
-   Knowledge graph
-   Graph database
-   RAG + agents
-   Automated job scraping
-   Automated job applications
-   Gmail integration
-   Calendar integration
-   Complex authentication
-   Microservices
-   Kubernetes
-   elaborate analytics
-   complicated frontend
-   unnecessary LangChain/LangGraph abstractions

If a feature does not improve retrieval quality, reliability,
evaluation, or demonstrable engineering skill, question whether it
belongs.

------------------------------------------------------------------------

# 23. Definition of Done

Project 1 is complete when:

-   [ ] User can upload PDF/DOCX/TXT documents.
-   [ ] Documents are parsed successfully.
-   [ ] Chunks preserve useful metadata.
-   [ ] Semantic retrieval works.
-   [ ] BM25 retrieval works.
-   [ ] Hybrid retrieval works.
-   [ ] Reranking works.
-   [ ] LLM generates grounded answers.
-   [ ] Answers contain source citations.
-   [ ] System can say "I don't know" when evidence is insufficient.
-   [ ] Evaluation dataset exists.
-   [ ] Retrieval metrics are calculated.
-   [ ] Baseline vs hybrid vs reranker results are documented.
-   [ ] Important failure cases are handled.
-   [ ] Tests pass.
-   [ ] Project is documented.
-   [ ] Project can be run by another developer.
-   [ ] Project is deployed or has a reproducible local/Docker setup.

------------------------------------------------------------------------

# 24. Portfolio Presentation

The GitHub README should emphasize:

## Problem

Basic document Q&A systems often retrieve irrelevant evidence and
provide little evidence that retrieval actually works.

## Solution

A document intelligence system combining lexical retrieval, semantic
retrieval, hybrid ranking, reranking, grounded generation, citations,
and evaluation.

## Engineering Focus

-   Retrieval quality
-   Reliability
-   Evaluation
-   Grounding
-   Explainability

## Results

Show actual experimental results.

Example:

``` text
Vector Retrieval       → Recall@5: XX%
BM25                   → Recall@5: XX%
Hybrid                 → Recall@5: XX%
Hybrid + Reranker      → Recall@5: XX%
```

Again: these values must come from the project's real evaluation.

------------------------------------------------------------------------

# 25. Interview Questions This Project Should Prepare You For

You should be able to explain:

### Retrieval

-   What is semantic search?
-   What is BM25?
-   Why combine lexical and semantic retrieval?
-   What is reranking?
-   Why retrieve top-N and rerank top-K?
-   How does chunk size affect retrieval?

### Embeddings

-   What is an embedding?
-   What does cosine similarity measure?
-   Why can semantic search miss exact terms?

### RAG

-   What is the difference between retrieval and generation?
-   Why does RAG reduce hallucination?
-   Can RAG still hallucinate?
-   What happens when the document doesn't contain the answer?

### Evaluation

-   Why is Recall@K useful?
-   What is MRR?
-   How did you create the evaluation dataset?
-   How do you know your new retrieval method is actually better?

### Engineering

-   How would this scale?
-   Where is latency introduced?
-   How would you cache embeddings?
-   How would you handle large documents?
-   What happens if the LLM API fails?

These questions are part of the value of the project.

------------------------------------------------------------------------

# 26. Final Project Philosophy

The project should follow one principle:

> **Do not make a simple retrieval task look complicated. Make a simple
> retrieval task technically reliable, measurable, and explainable.**

The goal is not to claim that we invented a new RAG architecture.

The goal is to demonstrate that we understand how to build, evaluate,
debug, and improve a real retrieval-based AI system.

That is the reason this project belongs in the portfolio.

------------------------------------------------------------------------

# 27. Recommended Build Order

``` text
1. Repository + FastAPI
2. Document parsing
3. Chunking + metadata
4. Vector retrieval
5. BM25
6. Hybrid retrieval
7. Reranking
8. Grounded LLM generation
9. Citations
10. Evaluation dataset
11. Retrieval experiments
12. Failure handling
13. Tests
14. Simple UI
15. Docker
16. Deployment
17. README + architecture diagram
18. Resume bullets
```

Build the system in this order.

Do not start with the UI.

Do not start with agents.

Do not start with a vector database because it sounds impressive.

Start with the **retrieval problem**, establish a baseline, improve it,
and measure the improvement.
