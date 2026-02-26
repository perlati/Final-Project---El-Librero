# backend/rag/chains.py

from collections import defaultdict
from typing import List, Optional, Dict, Any
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from backend.config import LLM_MODEL, BOOKS_COLLECTION, PROJECT_ROOT
from backend.vectorstore.store import get_vectorstore, get_all_documents
from backend.utils.text import normalize_title


def _normalize_title_for_grouping(title: str) -> str:
    """
    Normalize book title for deduplication.
    
    Strips production labels, normalizes case, removes extra whitespace.
    Used as dictionary key to group chunks from the same book.
    """
    if not title:
        return ""
    
    # Convert to lowercase for case-insensitive grouping
    normalized = title.lower().strip()
    
    # Remove common production labels
    production_labels = [
        "paper back", "paperback", "tripa final", "tripa", 
        "final", "draft", "revised", "edition"
    ]
    for label in production_labels:
        normalized = normalized.replace(label, "")
    
    # Remove year prefix patterns like "2018 " or "2018_"
    import re
    normalized = re.sub(r'^\d{4}[\s_-]+', '', normalized)
    
    # Normalize whitespace
    normalized = " ".join(normalized.split())
    
    return normalized


def _load_catalog_title_whitelist() -> set[str]:
    """
    Load normalized catalogue titles from data/books_metadata_llm.json.

    This whitelist is used as a guardrail so that only known catalogue
    titles are passed to the answer-shaping prompt.
    """
    metadata_path = Path(__file__).resolve().parents[2] / "data" / "books_metadata_llm.json"
    if not metadata_path.exists():
        return set()

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return set()

    titles = set()
    for file_name, book_data in (metadata or {}).items():
        if not isinstance(book_data, dict):
            continue

        title = (book_data.get("title") or "").strip()
        if title:
            normalized = _normalize_title_for_grouping(title)
            if normalized:
                titles.add(normalized)

        # Fallback: infer title from legacy filename format if needed.
        if not title and isinstance(file_name, str) and file_name.strip():
            stem = Path(file_name).stem
            parts = stem.split("_", 1)
            inferred = parts[1] if len(parts) > 1 else parts[0]
            inferred = inferred.replace("_", " ").strip()
            normalized = _normalize_title_for_grouping(inferred)
            if normalized:
                titles.add(normalized)

    return titles


CATALOG_TITLES = _load_catalog_title_whitelist()


def _load_books_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Return enriched metadata indexed by normalized title.

    Supports both list and dict JSON shapes.
    """
    metadata_path = PROJECT_ROOT / "data" / "books_metadata_llm.json"
    if not metadata_path.exists():
        return {}

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    by_title: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            nt = normalize_title(item.get("normalized_title") or item.get("title", "")).lower()
            if nt:
                by_title[nt] = item
    elif isinstance(data, dict):
        for _, item in data.items():
            if not isinstance(item, dict):
                continue
            nt = normalize_title(item.get("normalized_title") or item.get("title", "")).lower()
            if nt:
                by_title[nt] = item
    return by_title


BOOKS_METADATA = _load_books_metadata()


def _get_book_metadata_for_title(book_title: str) -> Dict[str, Any]:
    nt = normalize_title(book_title).lower()
    return BOOKS_METADATA.get(nt, {})


def _get_metadata_field(meta: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            txt = str(value).strip()
            if txt:
                return txt
    return ""


def _extract_metadata_view(book_title: str) -> Dict[str, str]:
    meta = _get_book_metadata_for_title(book_title)
    web_catalog = meta.get("web_catalog") if isinstance(meta.get("web_catalog"), dict) else {}

    title = _get_metadata_field(meta, "title") or normalize_title(book_title)
    author = (
        _get_metadata_field(meta, "author", "main_author", "book_author")
        or _get_metadata_field(web_catalog, "web_author")
        or "Autor no identificado"
    )
    year = (
        _get_metadata_field(meta, "year", "pub_year")
        or _get_metadata_field(web_catalog, "web_edition")
    )
    ficha = (
        _get_metadata_field(meta, "web_ficha", "ficha", "description")
        or _get_metadata_field(web_catalog, "web_description", "web_short_description")
    )
    tags_val = meta.get("tags") or meta.get("subjects") or web_catalog.get("web_tags", "")
    if isinstance(tags_val, list):
        tags = ", ".join(str(t).strip() for t in tags_val if str(t).strip())
    else:
        tags = str(tags_val).strip() if tags_val else ""

    return {
        "meta_title": title,
        "meta_author": author,
        "meta_year": year,
        "meta_ficha": ficha,
        "meta_tags": tags,
    }


def _filter_docs_by_catalog_whitelist(docs: List[Document]) -> List[Document]:
    """
    Keep only docs whose normalized book_title exists in CATALOG_TITLES.

    If whitelist loading fails (empty set), keep original docs to avoid
    hard-failing the retrieval pipeline.
    """
    if not docs or not CATALOG_TITLES:
        return docs

    filtered: List[Document] = []
    for doc in docs:
        metadata = doc.metadata or {}
        title = (metadata.get("book_title") or "").strip()
        normalized_title = _normalize_title_for_grouping(title)
        if normalized_title in CATALOG_TITLES:
            filtered.append(doc)

    return filtered


def _clean_author(author: str | None) -> str:
    """
    Clean author metadata to avoid showing filenames or production labels.
    
    Returns 'Autor no identificado' if:
    - Author is None or empty
    - Author contains file extensions (.indd, .pdf)
    - Author is ALL CAPS and suspiciously long (likely a filename)
    """
    if not author or not author.strip():
        return "Autor no identificado"
    
    author = author.strip()
    
    # Check for file extensions
    if ".indd" in author.lower() or ".pdf" in author.lower():
        return "Autor no identificado"
    
    # Check for ALL CAPS long strings (likely filenames)
    if author.isupper() and len(author) > 20:
        return "Autor no identificado"
    
    # Check for common production labels
    production_markers = ["paper back", "paperback", "tripa", "final"]
    if any(marker in author.lower() for marker in production_markers):
        return "Autor no identificado"
    
    return author


def docs_to_book_cards(docs: List[Document]) -> str:
    """
    Group retrieved chunks by book_title and create compact 'book cards'
    the LLM can reason over.

    Each card includes cleaned title, author, year and a few short snippets.
    Uses ONLY the cleaned metadata from LLM extraction, never reconstructs
    from filenames.
    
    Deduplicates books by normalizing titles (case-insensitive, strips production labels).
    """
    books = defaultdict(lambda: {
        "title": None,
        "author": None,
        "year": None,
        "file": None,
        "snippets": [],
    })

    for d in docs:
        md = d.metadata or {}
        # Use ONLY cleaned metadata from ingestion
        raw_title = md.get("book_title") or "Título desconocido"
        author = md.get("book_author")  # May be None
        year = md.get("pub_year")
        file = md.get("source")

        # Normalize title for grouping (deduplication key)
        normalized_key = _normalize_title_for_grouping(raw_title)
        
        entry = books[normalized_key]
        
        # Keep the FIRST title we see (usually cleanest)
        if entry["title"] is None:
            entry["title"] = raw_title
        
        # Prefer metadata with valid author (not corrupted)
        cleaned = _clean_author(author)
        if entry["author"] is None or (cleaned != "Autor no identificado" and entry["author"] == "Autor no identificado"):
            entry["author"] = cleaned
        
        # Prefer metadata with year
        if entry["year"] is None and year:
            entry["year"] = year
            
        # Keep first file path
        if entry["file"] is None:
            entry["file"] = file

        # Add a short snippet, limited length to avoid blowing the context
        text = d.page_content.strip().replace("\n", " ")
        if text and len(entry["snippets"]) < 4:  # cap snippets per book
            entry["snippets"].append(text[:350])

    lines = []
    for b in books.values():
        title = b["title"]
        author = b["author"]  # Already cleaned
        year = b["year"]
        file = b["file"]

        # Build clean header: Title (Year), Author
        header = f"Title: {title}"
        if year:
            header += f" ({year})"
        header += f", Author: {author}"

        # File path on separate line (not mixed with author)
        if file:
            header += f"\nFile: {file}"

        snippet_block = ""
        if b["snippets"]:
            snippet_lines = "\n".join(f"- {s}" for s in b["snippets"])
            snippet_block = f"\nSnippets:\n{snippet_lines}"

        lines.append(header + snippet_block)

    # This string is what goes into the 'books_context' variable in the prompt
    return "\n\n---\n\n".join(lines)


def _generate_query_variations(original_query: str, num_variations: int = 2) -> List[str]:
    """
    Generate query variations for multi-query retrieval.
    
    Uses LLM to generate semantically similar queries with:
    - Synonyms and related terms
    - Different phrasings
    - Topic expansions
    
    Args:
        original_query: The user's original question
        num_variations: Number of variations to generate (default: 2)
        
    Returns:
        List containing [original_query, variation1, variation2, ...]
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query expansion expert. Generate {num} alternative phrasings of the user's question using synonyms and related terms. Keep them concise and in the same language (Spanish). Return only the alternatives, one per line."),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": original_query, "num": num_variations})
    
    variations = [original_query]
    for line in response.content.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove numbering like "1. " or "- "
            cleaned = line.lstrip('0123456789.-) ').strip()
            if cleaned:
                variations.append(cleaned)
    
    return variations[:num_variations + 1]  # original + N variations


def get_books_retriever(filters: Optional[Dict[str, Any]] = None):
    """
    High-recall retriever for catalogue books using MMR for diversity.
    
    Args:
        filters: Optional metadata filters (e.g., {"pub_year": {"$gte": 2010}})
    """
    vs = get_vectorstore(BOOKS_COLLECTION)
    
    # Build filter: always include doc_type, merge with custom filters
    base_filter = {"doc_type": {"$eq": "catalog_book"}}
    if filters:
        # Combine base filter with custom filters using $and
        combined_filter = {"$and": [base_filter, filters]}
    else:
        combined_filter = base_filter
    
    retriever = vs.as_retriever(
        search_type="mmr",  # diversify results
        search_kwargs={
            "k": 40,          # higher recall so we see more of the catalogue
            "lambda_mult": 0.5,
            "filter": combined_filter,
        },
    )
    return retriever


def _format_docs(docs: List[Document]) -> str:
    """
    Format retrieved docs so the LLM clearly sees:

    - Book title (year) from the catalog
    - Author (if known)
    - Section type: body / bibliography
    """
    parts = []
    for d in docs:
        md = d.metadata or {}
        source = md.get("source", "unknown.pdf")
        book_title = md.get("book_title", source)
        pub_year = md.get("pub_year")
        book_author = md.get("book_author")
        page = md.get("page", "?")
        doc_type = md.get("doc_type", "catalog_book")
        section_type = md.get("section_type", "body")

        if pub_year:
            title_fmt = f"{book_title} ({pub_year})"
        else:
            title_fmt = book_title

        if book_author:
            author_fmt = book_author
        else:
            author_fmt = "Autor no identificado"

        header = (
            f"[DOC_TYPE: {doc_type} | LIBRO DEL CATÁLOGO: {title_fmt} | "
            f"AUTOR: {author_fmt} | ARCHIVO: {source} | PÁGINA: {page} | "
            f"SECCIÓN: {section_type}]"
        )
        parts.append(f"{header}\n{d.page_content}")

    return "\n\n---\n\n".join(parts)


def build_hybrid_retriever(filters: Optional[Dict[str, Any]] = None):
    """
    Ensemble retriever combining BM25 (keyword) + MMR (semantic) search.

    BM25 improves recall for exact names, events, and technical terms
    that the embedding space may under-weight.  The MMR retriever adds
    semantic recall for broad topics and synonymous phrasings.

    Weights: 30 % BM25 · 70 % MMR vector.

    Args:
        filters: Optional extra Chroma metadata filters (applied to the
                 vector leg only; BM25 is pre-filtered at doc load time).

    Returns:
        An EnsembleRetriever instance ready for .invoke(query).
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    # --- BM25 leg: keyword matching over all catalog chunks ---
    catalog_docs = get_all_documents(BOOKS_COLLECTION, doc_type="catalog_book")
    bm25_retriever = BM25Retriever.from_documents(catalog_docs, k=40)

    # --- MMR leg: semantic search with diversity ---
    mmr_retriever = get_books_retriever(filters=filters)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, mmr_retriever],
        weights=[0.3, 0.7],
    )


def _rerank_docs(docs: List[Document], query: str, top_k: int = 15) -> List[Document]:
    """
    LLM-based reranker: scores a candidate list in a single batch call
    and returns the top_k most relevant documents.

    The LLM is asked to return a comma-separated ranking of document
    numbers (1-based).  If parsing fails the original order is kept.

    Args:
        docs: Candidate documents from retrieval.
        query: The user's original question (used for relevance scoring).
        top_k: Maximum number of documents to return after reranking.

    Returns:
        Reranked list of up to top_k Document objects.
    """
    if not docs or len(docs) <= top_k:
        return docs

    candidates = docs[:20]  # cap to control cost

    numbered_snippets = "\n\n".join(
        f"[{i + 1}] {d.metadata.get('book_title', '?')} "
        f"({d.metadata.get('pub_year', '?')}): "
        f"{d.page_content[:250].strip().replace(chr(10), ' ')}"
        for i, d in enumerate(candidates)
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = (
        f"Eres un asistente editorial. Se hace la siguiente consulta:\n"
        f'"{query}"\n\n'
        f"A continuación hay {len(candidates)} fragmentos de libros del catálogo "
        f"numerados del 1 al {len(candidates)}.\n"
        f"Devuelve ÚNICAMENTE una lista de los {top_k} fragmentos más relevantes "
        f"para la consulta, ordenados de mayor a menor relevancia, como números "
        f"separados por comas (ejemplo: 3,1,7,2...).\n\n"
        f"Fragmentos:\n{numbered_snippets}"
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        # Parse "3,1,7,2" → 0-indexed list
        indices = [
            int(x.strip()) - 1
            for x in raw.split(",")
            if x.strip().lstrip("-").isdigit()
        ]
        valid = [i for i in indices if 0 <= i < len(candidates)]
        reranked = [candidates[i] for i in valid[:top_k]]
        # Fill any gap with remaining docs not yet included
        seen = set(valid[:top_k])
        for i in range(len(candidates)):
            if len(reranked) >= top_k:
                break
            if i not in seen:
                reranked.append(candidates[i])
        return reranked
    except Exception:
        return docs[:top_k]


def build_books_rag_chain(
    use_multi_query: bool = False,
    use_hybrid: bool = False,
    use_reranking: bool = False,
    filters: Optional[Dict[str, Any]] = None,
):
    """
    Build a comprehensive RAG chain over the books collection.

    Uses book-card aggregation to give the LLM a structured view of all
    relevant catalogue books and produce comprehensive answers.

    Retrieval pipeline (stages applied in order when enabled):
      1. **Hybrid** (use_hybrid=True): EnsembleRetriever combining BM25
         keyword search (30 %) with MMR semantic search (70 %).  Falls
         back to MMR-only when False.
      2. **Multi-query** (use_multi_query=True): generates 2 query
         variations with the LLM, merges and deduplicates results.
      3. **Reranking** (use_reranking=True): LLM scores the top-20
         candidate chunks in a single batch call and reorders them.

    Args:
        use_multi_query: Generate query variations for broader recall.
        use_hybrid: Use BM25 + MMR ensemble retrieval.
        use_reranking: Apply LLM reranking after retrieval.
        filters: Optional Chroma metadata filters applied to the vector
                 retriever (e.g. ``{"pub_year": {"$gte": 2010}}``).

    Input:  user question (string)
    Output: answer (string) listing all relevant books from the catalogue.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Eres El Librero, asistente editorial interno de Editorial Dahbar.\n"
                    "Solo puedes responder usando libros del catálogo propio.\n\n"
                    "Recibirás una lista llamada 'Catalogue entries'. Cada entrada corresponde a UN libro del catálogo\n"
                    "(title, year, author, file, snippets) y proviene de documentos con doc_type='catalog_book'.\n\n"
                    "Objetivo: dar una respuesta completa y confiable sobre el catálogo.\n\n"
                    "Reglas obligatorias:\n"
                    "1) Recorre TODAS las entradas y trata de identificar TODOS los libros claramente relacionados con la pregunta,\n"
                    "   no solo 1–2 ejemplos.\n"
                    "2) Lista TODOS los libros del catálogo que veas directamente relacionados, hasta un máximo de 15.\n"
                    "3) Para cada libro usa EXACTAMENTE este formato:\n"
                    "   - Título (Año), Autor\n"
                    "     Una sola oración explicando por qué es relevante.\n"
                    "4) Si la relación es débil o incierta, dilo explícitamente en la explicación (por ejemplo: 'relación parcial' o 'evidencia limitada').\n"
                    "5) NO inventes títulos, autores ni años.\n"
                    "6) NO menciones ninguna obra que no esté en 'Catalogue entries'.\n"
                    "7) NO menciones trabajos citados en bibliografías ni obras externas;\n"
                    "   incluso si aparecen en snippets, ignóralas si no son del catálogo (doc_type='catalog_book').\n"
                    "8) Si no hay libros claramente relevantes, dilo explícitamente.\n\n"
                    "Responde siempre en español."
                ),
            ),
            (
                "human",
                "User question:\n{question}\n\n"
                "Catalogue entries:\n{books_context}"
            ),
        ]
    )

    # Pipeline:
    #   1) hybrid or MMR retriever gets chunk-level docs
    #   2) optional multi-query merges results from query variations
    #   3) optional LLM reranking reorders by relevance
    #   4) docs_to_book_cards groups chunks into per-book entries
    #   5) prompt + LLM produce the final answer
    def _retrieve_and_group(question: str):
        # Step 1: choose retriever
        retriever = (
            build_hybrid_retriever(filters=filters)
            if use_hybrid
            else get_books_retriever(filters=filters)
        )

        # Step 2: retrieve (with optional multi-query expansion)
        if use_multi_query:
            queries = _generate_query_variations(question, num_variations=2)
            all_docs: List[Document] = []
            seen_ids: set = set()
            for q in queries:
                for doc in retriever.invoke(q):
                    doc_id = hash(doc.page_content)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
        else:
            all_docs = retriever.invoke(question)

        # Step 3: LLM reranking (optional)
        if use_reranking:
            all_docs = _rerank_docs(all_docs, question, top_k=15)

        # Step 4: strict whitelist guard against non-catalog artifacts
        all_docs = _filter_docs_by_catalog_whitelist(all_docs)

        # Step 5: group into book cards (cap at 40 chunks to avoid overflow)
        return docs_to_book_cards(all_docs[:40])

    rag_chain = (
        {
            "books_context": _retrieve_and_group,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | (lambda x: x.content)
    )

    return rag_chain


def build_book_summary_chain():
    """Build a one-book summary chain that returns exactly 10 bullet points."""

    vs = get_vectorstore(BOOKS_COLLECTION)

    def _get_book_docs(book_title: str) -> List[Document]:
        """
        Retrieve chunks only for the given book_title.
        Uses semantic search to find the most relevant book chunks.
        """
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 20,
                "filter": {"doc_type": "catalog_book"},
            },
        )
        all_docs = retriever.invoke(book_title)
        
        if not all_docs:
            return []
        
        book_counts = {}
        for d in all_docs[:10]:  # Look at top 10 results
            title = d.metadata.get("book_title", "")
            if title:
                book_counts[title] = book_counts.get(title, 0) + 1
        
        if not book_counts:
            return []
        
        target_book = max(book_counts, key=book_counts.get)
        
        return [d for d in all_docs if d.metadata.get("book_title") == target_book]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Eres El Librero, asistente editorial interno de Editorial Dahbar.\n"
                    "Debes responder únicamente usando el contexto suministrado.\n\n"
                    "Tarea: resumir un solo libro en EXACTAMENTE 10 viñetas.\n"
                    "Reglas obligatorias:\n"
                    "- Escribe en español.\n"
                    "- Usa solo información de este libro y de este contexto.\n"
                    "- No inventes datos, citas, autores ni hechos.\n"
                    "- No mezcles información de otros libros.\n"
                    "- Devuelve EXACTAMENTE 10 viñetas en formato Markdown usando '- '.\n"
                    "- Cada viñeta debe ser concreta y breve (1 oración).\n"
                    "- Si no hay contexto suficiente, di explícitamente: 'No encontré contenido suficiente para resumir este libro.'"
                ),
            ),
            (
                "human",
                "Título del libro: {book_title}\n"
                "Pregunta del usuario: {question}\n\n"
                "Contexto del libro:\n{context}\n\n"
                "Resume este libro en 10 viñetas."
            ),
        ]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def _with_context(payload: Dict[str, str]):
        book_title = (payload.get("book_title") or "").strip()
        question = (payload.get("question") or "Resume este libro en 10 viñetas.").strip()

        docs = _get_book_docs(book_title)
        if not docs:
            return {
                "book_title": book_title,
                "question": question,
                "context": "NO_CONTENT",
            }
        joined = "\n\n---\n\n".join(d.page_content for d in docs)
        return {
            "book_title": book_title,
            "question": question,
            "context": joined,
        }

    chain = (
        RunnablePassthrough()
        | _with_context
        | prompt
        | llm
        | (lambda x: x.content)
    )

    return chain


def build_deep_insight_chain():
    """
    Build a deep-insight chain for one catalogue book.

    Uses three sources when available:
    - Book chunks retrieved semantically (single-book focus)
    - Enriched metadata (title/author/year/tags/web ficha)
    - User focus question
    """
    vs = get_vectorstore(BOOKS_COLLECTION)

    def _get_book_docs(book_title: str) -> List[Document]:
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 30,
                "filter": {"doc_type": "catalog_book"},
            },
        )
        all_docs = retriever.invoke(book_title)
        if not all_docs:
            return []

        target_norm = normalize_title(book_title).lower()

        # Prefer exact normalized-title matches if present.
        exact_docs = [
            doc for doc in all_docs
            if normalize_title((doc.metadata or {}).get("book_title", "")).lower() == target_norm
        ]
        if exact_docs:
            return exact_docs

        # Fallback: most frequent title among top results.
        counts: Dict[str, int] = {}
        for doc in all_docs[:12]:
            title = ((doc.metadata or {}).get("book_title") or "").strip()
            if title:
                counts[title] = counts.get(title, 0) + 1

        if not counts:
            return []

        selected = max(counts.items(), key=lambda item: item[1])[0]
        return [doc for doc in all_docs if ((doc.metadata or {}).get("book_title") or "").strip() == selected]

    def _build_context(docs: List[Document]) -> str:
        if not docs:
            return "NO_CONTENT"

        parts: List[str] = []
        for doc in docs:
            metadata = doc.metadata or {}
            page = metadata.get("page", "?")
            section = metadata.get("section_type", "body")
            parts.append(f"[Página: {page} | Sección: {section}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Eres El Librero, lector profesional y editor senior.\n\n"
                    "Tu tarea es hacer un ANÁLISIS PROFUNDO de un solo libro del catálogo, usando:\n"
                    "- Fragmentos del propio libro (contexto)\n"
                    "- La ficha editorial / metadata cuando esté disponible\n\n"
                    "Objetivo:\n"
                    "Producir un informe que un editor pueda usar para vender el libro, posicionarlo\n"
                    "en el catálogo y recomendarlo a lectores y periodistas.\n\n"
                    "Reglas:\n"
                    "- Habla solo de UN libro (el indicado).\n"
                    "- No inventes datos biográficos ni hechos no respaldados por contexto/metadata.\n"
                    "- Si algo es conjetura, márcalo como interpretación.\n"
                    "- Escribe SIEMPRE en español, con tono profesional y claro."
                ),
            ),
            (
                "human",
                "Libro a analizar: \"{book_title}\"\n\n"
                "Pregunta o foco del usuario:\n{question}\n\n"
                "Metadata disponible:\n"
                "Título: {meta_title}\n"
                "Autor(es): {meta_author}\n"
                "Año: {meta_year}\n"
                "Tags: {meta_tags}\n"
                "Ficha web / descripción:\n{meta_ficha}\n\n"
                "Fragmentos del libro:\n{context}\n\n"
                "Ahora redacta un informe de lectura estructurado con este esquema:\n\n"
                "1. Tesis central del libro\n"
                "   - 2–3 frases que capturen el corazón de la obra.\n\n"
                "2. 4–6 ideas clave\n"
                "   - Cada viñeta: idea + breve explicación.\n"
                "   - Si es ensayo, subraya los argumentos más fuertes.\n"
                "   - Si es crónica o narrativa, subraya los hilos narrativos principales.\n\n"
                "3. Estructura y recorrido\n"
                "   - ¿Cómo se organiza el libro?\n"
                "   - ¿Qué tipo de lector necesita?\n\n"
                "4. Temas y marcos conceptuales\n"
                "   - Temas políticos, económicos, culturales o emocionales.\n"
                "   - Autores o teorías que aparecen o dialogan claramente con el texto.\n\n"
                "5. Valor editorial\n"
                "   - ¿Por qué es relevante para la editorial?\n"
                "   - ¿En qué conversaciones públicas entra?\n"
                "   - 2–3 ángulos para notas de prensa o presentaciones.\n\n"
                "Sé preciso y profundo, apoyándote en contexto y ficha.\n"
                "Si algún punto no se puede responder con seguridad, dilo explícitamente."
            ),
        ]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def _enrich_payload(payload: Dict[str, str]) -> Dict[str, str]:
        book_title = (payload.get("book_title") or "").strip()
        question = (payload.get("question") or "Dame un informe de lectura completo de este libro.").strip()

        docs = _get_book_docs(book_title)
        meta_view = _extract_metadata_view(book_title)
        return {
            "book_title": book_title,
            "question": question,
            "context": _build_context(docs),
            **meta_view,
        }

    chain = (
        RunnablePassthrough()
        | _enrich_payload
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def build_summarize_book_chain():
    """Backward-compatible alias."""
    return build_book_summary_chain()
