# backend/vectorstore/store.py

"""
Chroma vectorstore access layer.

Provides a single get_vectorstore() factory used throughout the project,
plus helpers for ingestion (add_documents_to_collection) and legacy
retriever access (get_retriever).

All collections share the same local persistence directory (VECTORSTORE_DIR)
and the same embedding model (EMBEDDING_MODEL), both configured in
backend/config.py.
"""

from typing import List

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from backend.config import VECTORSTORE_DIR, EMBEDDING_MODEL


MAX_EMBED_CHARS = 4000


def _make_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance for the configured embedding model."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore(collection_name: str) -> Chroma:
    """
    Open (or create) a persistent Chroma collection.

    Args:
        collection_name: Name of the Chroma collection (e.g. 'books_fulltext').

    Returns:
        A LangChain Chroma vectorstore ready for similarity/MMR search.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=_make_embeddings(),
        persist_directory=VECTORSTORE_DIR,
    )


def add_documents_to_collection(documents: List[Document], collection_name: str) -> int:
    """
    Add document chunks to a Chroma collection.

    Creates the collection if it does not exist. Chroma auto-persists on every
    write, so no explicit .persist() call is needed.

    Args:
        documents: List of LangChain Document objects (text + metadata).
        collection_name: Target Chroma collection name.

    Returns:
        Number of documents added.
    """
    vs = get_vectorstore(collection_name)

    # OpenAI embeddings have per-request token limits and per-input context
    # limits. Sanitize oversized texts and upload in batches to keep requests
    # safe and stable.
    sanitized_docs: List[Document] = []
    for doc in documents:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        if len(text) > MAX_EMBED_CHARS:
            text = text[:MAX_EMBED_CHARS]
        sanitized_docs.append(Document(page_content=text, metadata=doc.metadata or {}))

    def _safe_add(batch_docs: List[Document]) -> int:
        if not batch_docs:
            return 0
        try:
            vs.add_documents(batch_docs)
            return len(batch_docs)
        except Exception:
            added_local = 0
            # Fallback for edge cases where a single chunk still violates
            # embedding limits after initial sanitization.
            for item in batch_docs:
                text = (item.page_content or "").strip()
                if not text:
                    continue
                for max_len in (4000, 2000, 1000, 500):
                    try:
                        candidate = Document(
                            page_content=text[:max_len],
                            metadata=item.metadata or {},
                        )
                        vs.add_documents([candidate])
                        added_local += 1
                        break
                    except Exception:
                        continue
            return added_local

    batch_size = 64
    added = 0
    for start in range(0, len(sanitized_docs), batch_size):
        batch = sanitized_docs[start:start + batch_size]
        if batch:
            added += _safe_add(batch)

    return added


def get_all_documents(collection_name: str, doc_type: str | None = None) -> List[Document]:
    """
    Load all documents from a Chroma collection into memory.

    Used to build a BM25 index for hybrid retrieval.  Only call this at
    startup — it is a full collection scan and should not be called per
    query.

    Args:
        collection_name: Chroma collection to scan.
        doc_type: If set, keep only chunks whose ``doc_type`` metadata
                  field equals this value (e.g. ``"catalog_book"``).

    Returns:
        List of LangChain Document objects.
    """
    vs = get_vectorstore(collection_name)
    where = {"doc_type": {"$eq": doc_type}} if doc_type else None

    kwargs: dict = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where

    result = vs._collection.get(**kwargs)

    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        if text:  # skip empty chunks
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def get_retriever(collection_name: str, search_k: int = 6):
    """
    Return a simple similarity-search retriever (legacy helper).

    Superseded by get_books_retriever() in backend/rag/chains.py for active
    production use, but kept for backwards compatibility with
    backend/rag/retrievers.py.

    Args:
        collection_name: Chroma collection to query.
        search_k: Number of documents to retrieve per query.

    Returns:
        A LangChain retriever.
    """
    vs = get_vectorstore(collection_name)
    return vs.as_retriever(search_kwargs={"k": search_k})
