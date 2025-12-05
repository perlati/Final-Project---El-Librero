# backend/rag/retrievers.py

from backend.vectorstore.store import get_retriever
from backend.config import BOOKS_COLLECTION


def get_books_retriever(search_k: int = 6):
    """
    Return a retriever over the books collection.
    """
    return get_retriever(
        collection_name=BOOKS_COLLECTION,
        search_k=search_k,
    )
