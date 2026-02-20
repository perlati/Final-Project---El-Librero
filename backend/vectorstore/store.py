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
    vs.add_documents(documents)
    return len(documents)


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
