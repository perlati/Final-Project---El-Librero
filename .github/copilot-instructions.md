# EditorialCopilot – AI Coding Instructions

## Project Overview
EditorialCopilot is a **RAG-based AI assistant** for a Latin American publishing house (Editorial Dahbar) to query their book catalogue. Built with LangChain + OpenAI (gpt-4o-mini) + Chroma vector store, prioritizing **cost efficiency** for a small team (~5 users).

**Phase 1 (Current)**: Book catalogue Q&A from PDFs  
**Future Phases**: Contracts (Phase 2), Media/reviews (Phase 3), External book recommendations (Phase 4)

## Architecture

### Core Data Flow
1. **Ingestion**: `backend/ingestion/` loads PDFs from `data/books/`, enriches with LLM-extracted metadata from `data/books_metadata_llm.json`, chunks text (1200 chars, 200 overlap), stores in Chroma (`vectorstore/`)
2. **Retrieval**: User query → `backend/agents/editorial_agent.py` selects tool → tool invokes RAG chain → chain retrieves chunks → formats context → LLM generates answer
3. **UI**: Gradio chat interface (`app/gradio_app.py`) with conversation history

### Key Components
- **Agent**: `backend/agents/editorial_agent.py` – **simple keyword-based router** (NOT AgentExecutor)
  - Analyzes question patterns to select appropriate tool
  - No LangChain AgentExecutor/initialize_agent (removed for LangChain 1.0+ compatibility)
- **Tools**: `backend/agents/tools.py` – wrapped RAG chains:
  - `search_books`: Catalogue-wide topic search (k=40 MMR retrieval)
  - `summarize_book`: Single-book 3-point summary
  - `search_media`/`search_contracts`/`recommend_external_books` (stubs for future phases)
- **RAG Chains**: `backend/rag/chains.py` – two specialized chains:
  - **Catalogue search**: `build_books_rag_chain()` uses book cards aggregation
  - **Book summary**: `build_summarize_book_chain()` retrieves 20 chunks from one book
- **Metadata**: `backend/ingestion/extract_metadata_llm.py` uses GPT to extract title/author/year/subjects from first 6 pages

### Critical Distinctions
**Catalog books vs. external references**: PDFs contain bibliographies citing external works. The system uses:
- Metadata headers: `[DOC_TYPE: catalog_book | LIBRO DEL CATÁLOGO: Title (Year) | AUTOR: Name | SECCIÓN: body/bibliography]`
- Prompts explicitly instruct LLM to distinguish catalog books (in headers) from cited works (in bibliography sections)

## Development Workflows

### Running the App
```bash
# From project root
python app/gradio_app.py
# Opens Gradio UI at http://127.0.0.1:7860
```

### Ingesting Books
```bash
# Add PDFs to data/books/, then:
python -m backend.ingestion.ingest_books
# Loads PDFs, merges metadata from data/books_metadata_llm.json, stores in vectorstore/
```

### Extracting Metadata (Optional)
```bash
# If adding new books without metadata:
python -m backend.ingestion.extract_metadata_llm
# Analyzes first 6 pages of each PDF, outputs to data/books_metadata_llm.json
```

### Evaluation
```bash
python -m backend.evaluation.run_eval
# Runs questions from backend/evaluation/eval_questions.json through agent
```

## Project-Specific Conventions

### Agent Architecture (NO AgentExecutor)
**CRITICAL**: This project does NOT use LangChain's AgentExecutor framework due to LangChain 1.0+ breaking changes. Instead:
- **Simple routing**: `_select_tool_and_input()` in `editorial_agent.py` uses keyword patterns
- **Tools as plain functions**: Decorated with `@tool` from `langchain_core.tools`
- **Direct invocation**: `tool.invoke(input)` called directly, no executor wrapper

### Tool Selection Patterns
```python
# In editorial_agent.py
if "ideas centrales" in query or "resume" in query:
    tool = summarize_book
    input = extracted_title  # Uses regex to extract from query
elif "libros del catálogo" in query:
    tool = search_books
    input = query
```

### RAG Chain Types
1. **Book Cards Chain** (`search_books`):
   - Retrieves k=40 chunks with MMR
   - Groups by `book_title` using `docs_to_book_cards()`
   - LLM sees: "Title (Year), Author\nSnippets:\n- snippet1\n- snippet2..."
   - Returns up to 15 relevant books

2. **Summarize Chain** (`summarize_book`):
   - Semantic search with book title as query
   - Finds most common book in top 10 results (fuzzy matching)
   - Retrieves 20 chunks from that book
   - Returns exactly 3 bullet points

### Metadata Management
- **Dual-source metadata**: Combines LLM-extracted metadata (`data/books_metadata_llm.json`) with filename heuristics (e.g., `2018_Venezuela_en_el_nudo_gordiano_Paper_Back.pdf` → year=2018)
- **Title normalization**: Strips production artifacts ("Paper Back", "Tripa Final") via `normalize_catalog_title()` in `backend/ingestion/pdf_loader.py`
- **Bibliography detection**: `is_bibliography_page()` heuristic marks pages with citation patterns, stored as `section_type` metadata

### Prompt Engineering Patterns
All prompts in `backend/rag/chains.py` follow this structure:
1. Role definition: "Asistente editorial para Editorial Dahbar"
2. Metadata format explanation with examples
3. Explicit rules for catalog vs. external distinction
4. Citation requirements: always mention book title + page
5. Handling missing info: "di explícitamente que no la encuentras"

### Tool Selection Logic
`_select_tool()` in `editorial_agent.py` uses keyword matching:
- "contrato", "derechos" → `search_contracts` (stub)
- "entrevista", "reseña", "prensa" → `search_media` (stub)
- "recomienda", "externo" → `recommend_external_books` (stub)
- Default → `search_books` (active RAG)

Sub-dispatch within `search_books`: `_is_catalog_listing_query()` detects "¿Qué libros...?" patterns to switch between generic RAG and catalog listing chains.

### Configuration
`backend/config.py` centralizes:
- Model: `gpt-4o-mini` (cost optimization)
- Embeddings: `text-embedding-3-small`
- Collections: `BOOKS_COLLECTION = "books_fulltext"` (Chroma)
- Paths: `data/books/`, `vectorstore/`

Requires `.env` with `OPENAI_API_KEY`, optional `LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT`.

## Code Patterns

### Document Chunking
```python
# backend/ingestion/pdf_loader.py
RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
```
**Why 1200/200**: Balances context window vs. retrieval precision for Spanish-language books.

### Metadata Propagation
All chunks inherit per-book metadata (title, author, year, publisher, subjects, document_type) from `load_single_book()`. Critical for citation tracking.

### History Handling
`_history_to_text()` in `editorial_agent.py` normalizes Gradio's variable history formats (tuples/dicts) to text for LLM context.

### Error Handling
Gradio UI wraps agent calls with try/except, returning Spanish error messages prefixed with "Ocurrió un error".

## Common Tasks

### Adding a New RAG Tool
1. Define tool function in `backend/agents/tools.py` with `@tool` decorator
2. Add keyword patterns to `_select_tool()` in `editorial_agent.py`
3. Create chain builder in `backend/rag/chains.py` (see `build_books_rag_chain()` pattern)
4. Update Gradio UI radio options if needed

### Modifying Prompts
Edit templates in `backend/rag/chains.py`. Key sections:
- `<contexto>` block always uses `{context}` from retriever
- Rules section defines catalog vs. external logic
- Response format instructions (Markdown structure for listings)

### Debugging Retrieval
Check `_format_docs()` output in `chains.py` – metadata headers show what LLM sees. Common issues:
- Missing `book_title`: Check `data/books_metadata_llm.json` or filename heuristic
- Wrong author: LLM may extract editor/photographer – review extraction prompt in `extract_metadata_llm.py`

### Testing Specific Questions
Add to `backend/evaluation/eval_questions.json`, run `python -m backend.evaluation.run_eval`.

## Gotchas

- **Vectorstore persistence**: Chroma auto-persists to `vectorstore/`. Delete `vectorstore/` folder to reset.
- **Metadata JSON format**: `data/books_metadata_llm.json` keys are PDF filenames (e.g., `"2011_Aprender_a_ser_padres_Paper_Back.pdf"`).
- **Spanish-first**: All prompts/UI in Spanish. English questions work but expect Spanish responses.
- **Cost tracking**: gpt-4o-mini + small embeddings model chosen for ~$10-50/month usage at 40 queries/day/user (see `requirements.md` assumptions).
- **History limits**: No explicit truncation – long conversations may hit context limits.

## File Organization
```
backend/
  agents/     – Agent orchestration + tools
  ingestion/  – PDF loading, chunking, metadata extraction
  rag/        – Chain builders, retrievers
  vectorstore/ – Chroma interface
  evaluation/ – Test questions
app/
  gradio_app.py – Main UI
data/
  books/               – PDF source files
  books_metadata_llm.json – LLM-extracted metadata
vectorstore/            – Chroma DB (not in git)
```

## References
- `requirements.md`: Full spec with phases, cost modeling, architectural comparisons
- `backend/rag/chains.py`: Template examples for catalog distinction logic
- `backend/ingestion/pdf_loader.py`: Metadata normalization patterns
