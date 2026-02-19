# CLAUDE.md — EditorialCopilot (El Librero)

AI assistant reference for the EditorialCopilot project. Read this before making any changes.

---

## Project Overview

**EditorialCopilot** is a RAG-based AI assistant for Editorial Dahbar, a Latin American publishing house. It lets editorial staff query a catalogue of 73 books (PDFs) using natural language in Spanish.

**Current phase**: Phase 1 — Book catalogue Q&A from PDFs
**Future phases**: Contracts (Phase 2), Media/reviews (Phase 3), External recommendations (Phase 4)

**Stack**: Python 3.11+ · LangChain · OpenAI `gpt-4o-mini` · Chroma (local vector DB) · Gradio

---

## Repository Structure

```
.
├── app/
│   ├── gradio_app.py          # Main Gradio web UI — primary entry point
│   └── main_app.py            # Legacy entry point (unused)
├── backend/
│   ├── config.py              # Centralized config: models, paths, collection names
│   ├── demo_rag.py            # CLI demo for quick RAG testing
│   ├── agents/
│   │   ├── editorial_agent.py # Keyword-based router + agent orchestration
│   │   └── tools.py           # LangChain @tool-decorated functions (5 tools)
│   ├── rag/
│   │   ├── chains.py          # RAG chain builders (all prompts live here)
│   │   └── retrievers.py      # Legacy retrievers (largely superseded by chains.py)
│   ├── ingestion/
│   │   ├── ingest_books.py    # Ingestion pipeline: PDF → Chroma
│   │   ├── pdf_loader.py      # PDF parsing, chunking, metadata normalization
│   │   └── extract_metadata_llm.py  # LLM-based metadata extraction from PDFs
│   └── evaluation/
│       ├── run_eval.py        # Evaluation runner
│       └── eval_questions.json  # 10 reference Q&A pairs
├── data/
│   ├── books/                 # PDF source files (73 books, NOT in git)
│   └── books_metadata_llm.json  # LLM-extracted metadata (37KB, IS in git)
├── vectorstore/               # Chroma DB (~2.6GB, NOT in git)
├── requirements.txt           # Python dependencies
├── requirements.md            # Full 400-line project specification
└── .github/
    └── copilot-instructions.md  # Supplementary AI coding instructions
```

---

## Environment Setup

### Prerequisites

- Python 3.11+ (3.12 recommended)
- An OpenAI API key

### Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (never commit it):

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=editorialcopilot
```

`backend/config.py` loads these automatically. LangSmith tracing activates only if `LANGCHAIN_API_KEY` is set.

---

## Development Commands

All commands run from the **project root**.

### Launch the web app

```bash
python app/gradio_app.py
# Opens at http://127.0.0.1:7860
```

### Ingest books into the vector store

```bash
# Prerequisite: PDFs must be in data/books/
python -m backend.ingestion.ingest_books
```

### Extract LLM metadata from new PDFs (optional)

```bash
# Run before ingest_books when adding new PDFs without metadata entries
python -m backend.ingestion.extract_metadata_llm
# Output: data/books_metadata_llm.json
```

### Run the evaluation suite

```bash
python -m backend.evaluation.run_eval
# Runs 10 questions from eval_questions.json through the agent
```

### Quick CLI demo

```bash
python -m backend.demo_rag "¿Qué libros hablan sobre identidad latinoamericana?"
# Also accepts stdin if no argument is given
```

---

## Architecture

### Data Flow

```
User query (Gradio UI)
    ↓
editorial_agent.py — keyword-based tool selection
    ↓
tools.py — @tool function invoked via tool.invoke(input)
    ↓
chains.py — RAG chain (retriever + prompt + LLM)
    ↓
Chroma vectorstore — MMR retrieval (k=40)
    ↓
LLM (gpt-4o-mini) — generates Spanish answer
    ↓
Gradio UI — renders Markdown response
```

### Agent: Keyword-Based Router (NOT AgentExecutor)

**Critical**: This project does **not** use `AgentExecutor` or `initialize_agent`. LangChain 1.0+ broke those APIs. Instead, `editorial_agent.py` implements a simple keyword router:

```python
# _select_tool_and_input() in editorial_agent.py
if "ideas centrales" in query or "resume" in query:
    return summarize_book, extracted_title
elif "contrato" in query or "derechos" in query:
    return search_contracts, query   # stub
else:
    return search_books, query       # default active tool
```

Tools are invoked directly: `tool.invoke(input_string)` — no wrapper.

### Active Tools (Phase 1)

| Tool | Function | Behavior |
|------|----------|----------|
| `search_books` | `tools.py:search_books` | MMR retrieval (k=40), groups chunks by book, returns up to 15 books |
| `summarize_book` | `tools.py:summarize_book` | Retrieves 20 chunks from one book, returns exactly 3 bullet points |

### Stub Tools (Future Phases)

| Tool | Phase | Status |
|------|-------|--------|
| `search_contracts` | Phase 2 | Returns placeholder message |
| `search_media` | Phase 3 | Returns placeholder message |
| `recommend_external_books` | Phase 4 | Returns placeholder message |

### RAG Chains (`backend/rag/chains.py`)

Two chain builders exist:

1. **`build_books_rag_chain()`** — used by `search_books`
   - Retrieves k=40 chunks with MMR (lambda=0.5)
   - Groups chunks by `book_title` via `docs_to_book_cards()`
   - LLM context format: `Title (Year), Author\nSnippets:\n- snippet1…`

2. **`build_summarize_book_chain()`** — used by `summarize_book`
   - Semantic search using book title as query
   - Finds most common book in top 10 results (fuzzy matching)
   - Returns exactly 3 bullet points

### Vector Store (Chroma)

- **Location**: `vectorstore/` (local, not in git)
- **Active collection**: `books_fulltext`
- **Chunking**: 1200 chars / 200 overlap (`RecursiveCharacterTextSplitter`)
- **Embeddings**: `text-embedding-3-small`
- **Reset**: Delete the `vectorstore/` folder and re-run `ingest_books.py`

---

## Key Conventions

### Configuration is centralized in `backend/config.py`

Never hardcode model names, paths, or collection names. Always import from `backend.config`:

```python
from backend.config import LLM_MODEL, EMBEDDING_MODEL, BOOKS_COLLECTION, VECTORSTORE_DIR
```

### Document Metadata Schema

Every chunk stored in Chroma carries:

| Field | Example | Notes |
|-------|---------|-------|
| `book_title` | `"Venezuela en el nudo gordiano"` | Normalized (no "Paper Back", etc.) |
| `book_author` | `"Carlos Romero"` | LLM-extracted |
| `pub_year` | `2018` | From filename heuristic or LLM |
| `source` | `"2018_Venezuela_en_el_nudo_gordiano_Paper_Back.pdf"` | Original filename |
| `page` | `12` | Approximate page number |
| `doc_type` | `"catalog_book"` | Distinguishes from future contract/media chunks |
| `section_type` | `"body"` or `"bibliography"` | Heuristic (see `is_bibliography_page()`) |

### Metadata Header Format in Prompts

Chunks passed to the LLM are prefixed with headers that help it distinguish catalogue books from external references cited in bibliographies:

```
[DOC_TYPE: catalog_book | LIBRO DEL CATÁLOGO: Title (Year) | AUTOR: Name | SECCIÓN: body/bibliography]
```

### Prompts are all in `backend/rag/chains.py`

All prompt templates live in `chains.py`. When editing prompts:
- Keep the role definition: "Asistente editorial para Editorial Dahbar"
- Keep explicit rules separating catalogue books (in headers) from cited works (in bibliography sections)
- Preserve citation instructions: always mention book title + page
- Keep the "di explícitamente que no la encuentras" fallback rule

### Language

**Everything is Spanish-first.** UI labels, prompts, system instructions, and error messages are in Spanish. English queries work but responses will be in Spanish.

### Error Handling

Gradio UI wraps agent calls with try/except. Errors surface as Spanish messages prefixed with "Ocurrió un error". Do not introduce silent failures.

### History Normalization

`_history_to_text()` in `editorial_agent.py` handles Gradio's inconsistent history formats (tuples vs. dicts). Use that helper; do not re-implement history formatting elsewhere.

---

## Adding Features

### Add a new RAG tool

1. Create a chain builder in `backend/rag/chains.py` (follow `build_books_rag_chain()` pattern)
2. Define a `@tool`-decorated function in `backend/agents/tools.py`
3. Add keyword routing to `_select_tool_and_input()` in `backend/agents/editorial_agent.py`
4. Update Gradio UI tool list in `app/gradio_app.py` if needed

### Modify retrieval behavior

- **k (chunks retrieved)**: Adjust `search_kwargs={"k": 40}` in `chains.py`
- **MMR diversity**: Adjust `lambda_mult` (0.0 = max diversity, 1.0 = max relevance)
- **Multi-query**: Enable/disable `MultiQueryRetriever` wrapping in chain builder

### Add new books

1. Place PDFs in `data/books/`
2. Optionally run `python -m backend.ingestion.extract_metadata_llm` to extract metadata
3. Run `python -m backend.ingestion.ingest_books` to re-ingest

### Add evaluation questions

Edit `backend/evaluation/eval_questions.json`:

```json
[
  {
    "question": "¿Qué libros hablan sobre migración?",
    "expected_keywords": ["migración", "Venezuela"]
  }
]
```

Then run `python -m backend.evaluation.run_eval`.

---

## Gotchas

- **Vectorstore must be built before first run.** It is excluded from git (~2.6GB). Run `ingest_books.py` first.
- **PDF files are not in git.** `data/books/` is gitignored. Obtain them separately.
- **No AgentExecutor.** Do not introduce `initialize_agent` or `AgentExecutor` — they broke in LangChain 1.0+.
- **Metadata JSON keys are PDF filenames.** `data/books_metadata_llm.json` is keyed by the exact filename (e.g., `"2011_Aprender_a_ser_padres_Paper_Back.pdf"`). Filename changes break metadata lookup.
- **Title normalization strips production artifacts.** "Paper Back", "Tripa Final", etc. are removed by `normalize_catalog_title()` in `pdf_loader.py`. Ensure new PDFs follow the `YEAR_Title_Words.pdf` naming convention.
- **No context truncation for chat history.** Long conversations may hit the LLM context window limit. This is a known gap.
- **Chroma auto-persists.** No explicit `.persist()` call needed. The DB writes on every add.
- **Cost target: ~$5–10/month** for 5 users at ~40 queries/day. Keep using `gpt-4o-mini` and `text-embedding-3-small`.

---

## Cost Model

| Component | Model | Est. Monthly Cost |
|-----------|-------|------------------|
| LLM inference | `gpt-4o-mini` | ~$3–7/month |
| Embeddings (new ingestion only) | `text-embedding-3-small` | < $0.50 one-time |
| Vector store | Chroma (local) | $0 |
| **Total** | | **~$5–10/month** |

Assumptions: 5 users, 40 queries/day/user, ~2K tokens/query.

---

## References

- `requirements.md` — Full specification with phases, functional requirements (FR-1 to FR-19), and cost modeling
- `backend/rag/chains.py` — All prompt templates and chain logic
- `backend/ingestion/pdf_loader.py` — Metadata normalization and bibliography detection
- `.github/copilot-instructions.md` — Supplementary AI coding patterns
- `data/books_metadata_llm.json` — Authoritative book metadata (title, author, year, subjects)
