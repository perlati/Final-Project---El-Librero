# backend/ingestion/ingest_books.py

"""
Ingest all book PDFs from data/books/ into the Chroma collection 'books_fulltext'.

Run from project root as:
    python -m backend.ingestion.ingest_books
"""

from backend.config import BOOKS_DIR, BOOKS_COLLECTION
from backend.ingestion.pdf_loader import load_and_chunk_books
from backend.vectorstore.store import add_documents_to_collection


def main():
    print("[INGEST] Starting book ingestion...")
    chunks = load_and_chunk_books(BOOKS_DIR)
    print(f"[INGEST] Got {len(chunks)} chunks. Uploading to vector store...")

    added = add_documents_to_collection(chunks, BOOKS_COLLECTION)
    print(f"[INGEST] Done. Added {added} chunks to collection '{BOOKS_COLLECTION}'.")


if __name__ == "__main__":
    main()
