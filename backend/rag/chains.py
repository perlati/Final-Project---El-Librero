# backend/rag/chains.py

from collections import defaultdict
from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from backend.config import LLM_MODEL, BOOKS_COLLECTION
from backend.vectorstore.store import get_vectorstore


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


def build_books_rag_chain(use_multi_query: bool = False, filters: Optional[Dict[str, Any]] = None):
    """
    Build a comprehensive RAG chain over the books collection.
    
    Uses book cards aggregation to show the LLM all relevant catalogue books,
    enabling comprehensive answers that list multiple relevant titles.

    Args:
        use_multi_query: If True, generate query variations for better recall
        filters: Optional metadata filters (e.g., {"pub_year": {"$gte": 2010}})

    Input: user question (string)
    Output: answer (string) listing all relevant books from the catalogue.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are El Librero, an internal assistant for a Latin American "
                    "publishing house (Editorial Dahbar). You only answer using the publisher's own "
                    "book catalogue.\n\n"
                    "You will receive a list of catalogue entries in 'Catalogue entries'. "
                    "Each entry describes ONE book in the catalogue (title, author, year, file, snippets).\n\n"
                    "Your task is to:\n"
                    "1) Carefully scan ALL the books listed in 'Catalogue entries'.\n"
                    "2) Identify EVERY book that is clearly relevant to the user's question.\n"
                    "3) Answer by listing ALL relevant catalogue books you find (not just 1–2), "
                    "up to a maximum of 15 titles, in this format:\n"
                    "   - Title (Year), Author — 1–2 sentence explanation of why this book is relevant.\n\n"
                    "If you truly find no relevant catalogue books, say so explicitly.\n"
                    "Do NOT hallucinate titles that are not in the catalogue entries you received.\n"
                    "Do NOT propose external or bibliographic books; only use the catalogue.\n\n"
                    "Always respond in Spanish."
                ),
            ),
            (
                "human",
                "User question:\n{question}\n\n"
                "Catalogue entries:\n{books_context}"
            ),
        ]
    )

    # 1) retriever gets chunk-level docs
    # 2) docs_to_book_cards groups them into per-book entries
    # 3) prompt + llm produce final answer
    def _retrieve_and_group(question: str):
        if use_multi_query:
            # Generate query variations and retrieve for each
            queries = _generate_query_variations(question, num_variations=2)
            all_docs = []
            seen_ids = set()
            
            retriever = get_books_retriever(filters=filters)
            for query in queries:
                docs = retriever.invoke(query)
                # Deduplicate by page_content hash
                for doc in docs:
                    doc_id = hash(doc.page_content)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            
            # Limit to top 40 to avoid context overflow
            return docs_to_book_cards(all_docs[:40])
        else:
            retriever = get_books_retriever(filters=filters)
            docs = retriever.invoke(question)
            return docs_to_book_cards(docs)

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


def build_summarize_book_chain():
    """
    RAG chain to summarize ONE specific book from the catalogue
    in 3 key ideas / bullet points.
    """

    vs = get_vectorstore(BOOKS_COLLECTION)

    def _get_book_docs(book_title: str) -> List[Document]:
        """
        Retrieve chunks only for the given book_title.
        Uses semantic search to find the most relevant book chunks.
        """
        # Use semantic search with doc_type filter only
        # This allows fuzzy/semantic matching of the title
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 20,
                "filter": {"doc_type": {"$eq": "catalog_book"}},
            },
        )
        # Search using the book title - will find semantically similar chunks
        all_docs = retriever.invoke(book_title)
        
        # Filter to keep only docs from the most relevant book
        # (the one that appears most in the top results)
        if not all_docs:
            return []
        
        # Count which book appears most in top results
        book_counts = {}
        for d in all_docs[:10]:  # Look at top 10 results
            title = d.metadata.get("book_title", "")
            if title:
                book_counts[title] = book_counts.get(title, 0) + 1
        
        if not book_counts:
            return []
        
        # Get the most common book title
        target_book = max(book_counts, key=book_counts.get)
        
        # Return all docs from that book
        return [d for d in all_docs if d.metadata.get("book_title") == target_book]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are El Librero, an internal assistant for a Latin American "
                    "publishing house. You ONLY answer using the publisher's own books.\n\n"
                    "You will receive snippets from ONE specific book, identified by its "
                    "catalogue title.\n\n"
                    "Your task:\n"
                    "1) Read the provided snippets carefully.\n"
                    "2) Identify the 3 most important ideas or themes of that book.\n"
                    "3) Answer in Spanish with EXACTLY 3 bullet points, each 1–3 sentences.\n\n"
                    "Be concrete and refer to the book's content, not meta information.\n"
                    "If you cannot find any content for that title, say clearly that you "
                    "cannot summarize it because there is no data."
                ),
            ),
            (
                "human",
                "Título del libro: {book_title}\n\n"
                "Fragmentos del libro:\n{context}\n\n"
                "Resume las 3 ideas centrales de este libro."
            ),
        ]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    def _with_context(book_title: str):
        docs = _get_book_docs(book_title)
        if not docs:
            # Let the prompt see an empty context, LLM will handle the 'no data' path
            return {"book_title": book_title, "context": "NO_CONTENT"}
        joined = "\n\n---\n\n".join(d.page_content for d in docs)
        return {"book_title": book_title, "context": joined}

    chain = (
        RunnablePassthrough()  # passes book_title through
        | _with_context
        | prompt
        | llm
        | (lambda x: x.content)
    )

    return chain
