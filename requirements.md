# EditorialCopilot – Requirements Document

## 1. Context

The client is a small Latin American publishing house (≈5 people) that wants an internal AI assistant to:

- Answer questions about its **book catalogue** (PDF archive).
- Later: answer questions about **covers and media**, **contracts**, and potentially external books that match its editorial line.

Key constraints:

- **Very low cost** (LATAM context, limited FX access, no large recurring SaaS bills).
- **Possible fully local deployment in the office** (on-prem machine or LAN-only server).
- Clear understanding of trade-offs between:
  1. OpenAI RAG tools (OpenAI-managed vector store / retrieval),
  2. Custom RAG built on OpenAI APIs,
  3. Fully **open-source** local stack.

The app is an internal tool, not public-facing.

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. Enable staff to query the publisher’s catalogue in natural language (Spanish first; English optional).
2. Provide answers grounded in **specific books/sections/pages**, with explicit citations.
3. Be cheap enough that 5 people can use it daily without worrying about API bills.
4. Have a deployment option that can run **locally in the office**.
5. Provide a technical/commercial **analysis** of three architectures:
   - OpenAI RAG tools (OpenAI-managed retrieval),
   - OpenAI API + self-managed RAG,
   - Fully open-source local solution.
6. Organize data integration in four phases:
   - Phase 1: Book catalogue (PDFs).
   - Phase 2: Covers & media related to books.
   - Phase 3: Contracts & rights.
   - Phase 4: External open-source book APIs to recommend titles aligned with the editorial.

### 2.2 Non-Goals

- Not a general-purpose public chatbot.
- No complex analytics dashboards beyond basic usage metrics.
- No full contract lifecycle management; only **QA** over existing contracts.
- No full document management system; relies on an external folder structure for PDFs.

---

## 3. Users & Usage

### 3.1 User Roles

- **Editor-in-chief**: conceptual queries (themes, positioning, overview).
- **Acquisitions editor**: checks catalogue to avoid overlaps; compares topics.
- **Rights manager**: future – rights and territory questions over contracts.
- **Marketing**: prepares blurbs, press notes, comparative titles.
- **Assistant/editorial staff**: quick fact checks, data gathering.

### 3.2 Usage assumptions (for cost modeling)

Working assumptions for 5 users:

- **Low usage**: 10 questions/day/user, 22 working days/month.
- **Medium usage (base case)**: 40 questions/day/user, 22 days/month.
- **High usage**: 100 questions/day/user, 22 days/month.

Average per query:

- ≈1,000 input tokens (user question + retrieved context).
- ≈500 output tokens (answer).

---

## 4. Scope & Data Phases

### Phase 1 – Book Catalogue (Internal PDFs) – MVP

**Scope**

- Ingest and index the **book catalogue**:
  - PDFs of published books (and optionally manuscripts).
  - Metadata: title, author, year, collection/series, topics, language, ISBN.

**Doc types / collections**

- Vector store collections:
  - `books_fulltext` – chunked full text of books.
- Metadata field `doc_type = "book"`.

**Main use cases**

- “List and summarize our books about X topic.”
- “What does author A argue about subject Y?”
- “Compare our books on the same theme.”

This is the **primary focus** of the bootcamp project.

---

### Phase 2 – Covers & Media Related to Books

**Scope**

- Extend the system to store **non-text material associated with books**, but still retrieve and answer using text:

  1. **Book covers**:
     - Store images plus short text metadata (title, author, collection, design notes).
     - RAG uses metadata text; image pixels are optional (no vision model required).

  2. **Media related to books**:
     - Author interviews (YouTube links, podcast transcripts).
     - TV/radio features converted to transcripts.
     - Press articles or reviews in PDF or converted from HTML.

**Doc types / collections**

- `media` – transcripts and press pieces.
- `book_covers` – text metadata for covers.
- Metadata field `related_book_id` linking media to internal books.

**Use cases**

- “Show me media mentions and interviews for this book.”
- “What did the author say about topic X in interviews?”
- “Which books have the strongest media presence on theme Y?”

For the project, a small number of media docs is sufficient to demonstrate the architecture.

---

### Phase 3 – Contracts & Rights

**Scope**

- Ingest and index **publishing contracts** and rights documents (internal, confidential).

**Doc types / collections**

- `contracts` – chunked contract text with metadata:
  - `doc_type = "contract"`,
  - `parties`, `territory`, `language_rights`, `print_run`, `start_date`, `end_date`.

**Use cases**

- “What are the reprint rights for this book in Spain?”
- “Does author X’s contract allow digital rights in Mexico?”
- “When do we recover rights for book Y?”

**Behavior**

- Contract QA must be conservative:
  - Always include citations.
  - Highlight uncertainties and recommend human legal review.
  - Explicitly state that the answer is not legal advice.

---

### Phase 4 – External Open-Source Book APIs (Editorial Recommendations)

**Scope**

- Connect to an **open-source or free book API** (e.g., Open Library or similar) to discover external books that fit the publisher’s editorial line.

**Doc types / collections**

- `external_books` – metadata from external APIs:
  - title, author, subjects/tags, year, language, etc.
- Map external subjects/tags to the publisher’s editorial axes.

**Use cases**

- “Which external books resemble our collection on topic X?”
- “Suggest acquisition opportunities that fit our editorial line.”
- “Show foreign comp titles for this proposal.”

**Implementation detail**

- Implement as a separate agent tool, e.g. `ExternalBookRecommenderTool`, that:
  - Receives a topic or tags,
  - Calls the external API,
  - Returns a ranked list of candidate titles.

---

## 5. Functional Requirements

### 5.1 Data ingestion & organization

FR-1. The system must ingest **PDF book files** from a configured directory (`data/books/`) or similar storage.

FR-2. Each book must have structured metadata:

- `doc_id`, `title`, `author`, `year`, `collection`, `language`, `ISBN`, `doc_type = "book"`, `tags`.

FR-3. The system must split each book into chunks (e.g. 1–2k characters with overlap), preserving:

- Original book reference and approximate **page number or section**.

FR-4. The system must store chunks + metadata in a **vector database** (e.g. Chroma) for semantic search.

FR-5. The ingestion pipeline must be **repeatable and idempotent**:
- Re-running ingestion on the same book should update or skip duplicates safely.

FR-6. The system must be ready (even if partially implemented) to ingest:

- Media transcripts (`doc_type = "media"`),
- Contracts (`doc_type = "contract"`),
- External metadata from book APIs (`doc_type = "external_book"`).

---

### 5.2 Question answering (RAG) – Text Only

FR-7. The system must accept **text** natural language questions (Spanish/English) and return grounded answers.

FR-8. For each answer, the system must provide:

- 1–3 cited passages showing:
  - book title,
  - approximate page/chapter,
  - snippet text.

FR-9. The RAG pipeline must:

- Retrieve relevant chunks from the vector store,
- Pass them and the question to an LLM with a custom prompt,
- Prefer “I don’t know” when context is insufficient or conflicting.

FR-10. The user must be able to **filter corpus**:

- `Books only`
- `Contracts only` (when Phase 3 data exists)
- `Media only` (when Phase 2 data exists)
- `All documents`

FR-11. The system must support **conversational follow-ups**:

- Maintain conversation history per session (e.g. “And in later books?”)
- Use memory to interpret pronouns and ellipsis.

---

### 5.3 Conversational web interface

FR-12. Implement a web UI (Streamlit, Gradio, or simple frontend) with:

- Chat-style interface.
- Text input box (no audio input).
- Corpus selector (dropdown or radio buttons).
- Display of:
  - Answer text,
  - Citations / sources panel (expandable).

FR-13. The UI must manage sessions for **5 parallel users** on the same instance (each with its own history).

FR-14. Provide a simple **usage view**:

- Total number of queries served (per day or since start).

---

### 5.4 Agents and tools (LangChain)

FR-15. The system must use **LangChain agents** to orchestrate tools, including at least:

- `SearchBooksTool`
- `SearchContractsTool` (stub or minimal)
- `SearchMediaTool` (stub or minimal)
- `ExternalBookRecommenderTool` (Phase 4, conceptual or minimal demo)

FR-16. The agent must decide which tool(s) to call based on:

- User question,
- Selected corpus filter from the UI (books / contracts / media / all).

FR-17. The agent must use **conversation memory** per session (e.g. buffer or summary memory).

---

### 5.5 LangSmith integration

FR-18. All major chains and agent calls must be traced in **LangSmith**.

FR-19. Provide a small evaluation set (10–20 Q&A pairs) and a script to run evaluation and log:

- Answer correctness (manually scored or noted),
- Citation presence,
- Latency.

---

## 6. Non-Functional Requirements

NFR-1. **Cost**: For base usage (5 users, medium scenario), the monthly server + API cost should ideally stay **under ~USD 20** in OpenAI-based configurations; the open-source configuration should have near-zero marginal cost once hardware is purchased.

NFR-2. **Latency**: For typical queries, 95th percentile response time under **7–8 seconds** on normal office broadband.

NFR-3. **Privacy**:

- Internal PDFs, contracts, and metadata must never be shared beyond the chosen providers.
- For on-prem deployments, all raw documents and embeddings must stay on the local machine.

NFR-4. **Deployability**:

- Must run on a single server (Linux or Windows) that can be placed in the office.
- A dockerized setup is preferred but not required for the bootcamp.

NFR-5. **Maintainability**:

- Clear folder structure and README.
- Ingestion can be re-run by a non-developer following written instructions.

NFR-6. **Extensibility**:

- Easy to add new document types and tools (e.g., press releases, new APIs) by extending ingestion scripts and metadata.

---

## 7. Architecture Overview (MVP)

- **Frontend:** Streamlit web app served either:
  - locally on a desktop, or
  - on an office server reachable by LAN.

- **Backend (Python):** LangChain-based orchestration of:
  - RAG pipeline (retrievers + LLM),
  - Agent + tools for books/contracts/media/external APIs,
  - Conversation memory.

- **Vector store:** Chroma (local folder) for MVP; abstraction layer to optionally swap to Pinecone.

- **LLM & Embeddings (OpenAI-based variant):**
  - LLM: GPT-4o mini (cost-efficient, strong quality).
  - Embeddings: `text-embedding-3-small`.

- **Open-source variant (future):**
  - Local LLM (e.g., Llama-3 or Mistral 7–8B) via Ollama/vLLM.
  - Local embedding models (`sentence-transformers`).
  - Same LangChain RAG/agent logic, but all models on-prem.

---

## 8. Deployment Options for the Frontend

### 8.1 F1 – Local Desktop App (Single Machine)

- Run `streamlit run app/main_app.py` on one machine.
- Only that machine can use the app via browser (`http://localhost:8501`).
- Zero hosting cost; simplest for development and solo use.

### 8.2 F2 – Local Office Server (LAN-Only)

- Deploy app + backend on a single “office server” (could be a reused PC).
- Expose it on LAN (`http://192.168.x.x:8501`).
- All 5 staff access it via browser.
- Zero cloud cost; documents and embeddings stay inside the office network.

### 8.3 F3 – Low-Cost Cloud (Optional Demo)

- Deploy on cheap/free cloud (Hugging Face Spaces, Render, etc.) mainly for:
  - Remote demos,
  - Bootcamp presentation.
- Not required for the real-world local deployment.

---

## 9. Architecture Options & Cost Comparison

The project will implement and compare three approaches:

1. **Option A – OpenAI RAG tools (OpenAI-managed retrieval)**
   - Store documents and embeddings with OpenAI.
   - Pay per query + storage.
   - Simpler infrastructure but less control and slightly higher recurring cost.

2. **Option B – Custom RAG on OpenAI APIs (Primary implementation)**
   - Use OpenAI only for:
     - Embeddings (cheap, one-off),
     - LLM calls for generation.
   - Host vector store (e.g., Chroma) locally.
   - Expected cost for 5 users at medium usage with GPT-4o mini:
     - **≈$1–6/month** total API cost.

3. **Option C – Fully open-source local stack**
   - Local vector DB + local LLM + local embeddings.
   - API cost ~0, but requires suitable hardware and more maintenance.
   - Strongest privacy; ideal long-term for a small LATAM publisher if they invest in a local server.

Option B will be the **main implemented architecture** for the bootcamp project, with A and C described and partially prototyped (where feasible) in documentation and slides.
