# backend/ingestion/pdf_loader.py

import os
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.config import BOOKS_DIR

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

import json
from pathlib import Path

LLM_METADATA_PATH = Path("data/books_metadata_llm.json")

JUNK_TITLE_PHRASES = [
    "paper back",
    "paperback",
    "tripa final",
    "tripa",
    "final",
    "tapa blanda",
    "edición",
    "edicion",
    "colección",
    "coleccion",
]


def normalize_catalog_title(raw: str | None) -> str:
    """
    Normalize book titles for catalog use.

    - Collapse spaces.
    - Remove technical / production phrases (Paper Back, Tripa Final, etc.).
    - Strip leftover punctuation.
    """
    if not raw:
        return ""

    t = " ".join(raw.split())  # collapse whitespace

    # Remove junk phrases word-by-word
    words = t.split()
    cleaned_words = []
    for w in words:
        lw = w.lower()
        if any(phrase in lw for phrase in JUNK_TITLE_PHRASES):
            continue
        cleaned_words.append(w)

    t = " ".join(cleaned_words).strip(" -_,.:;")
    return t


def list_pdf_files(base_dir: str = BOOKS_DIR) -> List[str]:
    """
    Return all PDF file paths under BOOKS_DIR (recursive).
    """
    pdf_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))
    return sorted(pdf_files)


import os
import re

JUNK_TITLE_TOKENS = {"paper", "back", "tripa", "final"}


def infer_book_title_and_year_from_filename(path: str) -> tuple[str | None, str | None]:
    fname = os.path.basename(path)
    # Strip extension
    name_no_ext = os.path.splitext(fname)[0]

    # Expect "YYYY_rest_of_title"
    parts = name_no_ext.split("_", 1)
    year = None
    rest = name_no_ext
    if len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 4:
        year = parts[0]
        rest = parts[1]

    # Replace underscores with spaces for a rough title
    raw_title = rest.replace("_", " ").strip()
    return raw_title, year



def is_bibliography_page(text: str) -> bool:
    """
    Heuristic detection of bibliography / references pages.

    We don't need perfection; goal is to mark pages that are *mostly* lists of citations
    so the LLM can distinguish 'cited works' from the main body of the catalog book.
    """
    lower = text.lower()

    # Strong clue: section headings
    if "bibliografía" in lower or "bibliografia" in lower:
        return True
    if "notas" in lower:
        return True
    if "referencias bibliográficas" in lower or "referencias" in lower:
        return True
    if "bibliography" in lower or "references" in lower:
        return True

    # Heuristic on structure: many short lines that look like "Apellido, Inicial. (Año)..."
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False

    citation_like = 0
    for l in lines:
        # Example patterns: "Carothers, T. (2002).", "Levitsky, S. y Way, L. (2010)..."
        if re.search(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+,\s*[A-Z]\.", l) and re.search(r"\(19\d{2}\)|\(20\d{2}\)", l):
            citation_like += 1

    if citation_like >= max(3, len(lines) // 4):
        # At least 3 lines AND roughly >=25% of lines look like citations
        return True

    return False


def load_single_book(path: str, llm_meta: dict | None = None) -> List[Document]:
    loader = PyPDFLoader(path)
    pages: List[Document] = loader.load()

    file_name = os.path.basename(path)

    meta_entry = (llm_meta or {}).get(file_name, {})
    llm_title = meta_entry.get("title") or None
    llm_subtitle = meta_entry.get("subtitle") or None
    llm_main_author = meta_entry.get("main_author") or None
    llm_year = meta_entry.get("year") or None  # optional, mostly diagnostic

    # NEW: filename year as canonical
    filename_title, filename_year = infer_book_title_and_year_from_filename(path)
    inferred_author = extract_book_author_from_pages(pages)

    # Title: LLM -> filename -> fallback
    raw_title = llm_title or filename_title
    book_title = normalize_catalog_title(raw_title)

    # Author: LLM -> heuristic
    book_author = llm_main_author or inferred_author

    # Year: filename first, then LLM (only if filename missing / weird)
    pub_year = filename_year or llm_year

    enriched_pages: List[Document] = []
    for d in pages:
        md = dict(d.metadata) if d.metadata else {}

        md["source"] = file_name
        md["book_id"] = os.path.splitext(file_name)[0]
        md["book_title"] = book_title
        md["book_author"] = book_author
        md["pub_year"] = pub_year
        md["doc_type"] = "catalog_book"
        md["section_type"] = "bibliography" if is_bibliography_page(d.page_content) else "body"

        # Optional: keep raw LLM year for debugging, but not used in answers
        md["pub_year_llm"] = llm_year

        # Extra fields (already discussed)
        md["publisher"] = meta_entry.get("publisher")
        md["document_type"] = meta_entry.get("document_type")
        subjects = meta_entry.get("subjects")
        if isinstance(subjects, list):
            md["subjects"] = ", ".join(s for s in subjects if isinstance(s, str))
        else:
            md["subjects"] = subjects

        enriched_pages.append(
            Document(page_content=d.page_content, metadata=md)
        )

    return enriched_pages


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split page-level documents into overlapping chunks, preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


from backend.config import BOOKS_DIR

def load_and_chunk_books(base_dir: str = BOOKS_DIR) -> List[Document]:
    pdf_files = list_pdf_files(base_dir)
    llm_meta = load_llm_metadata()
    all_chunks: List[Document] = []

    if not pdf_files:
        print(f"[INGEST] No PDF files found under {base_dir}")
        return []

    print(f"[INGEST] Found {len(pdf_files)} PDF files under {base_dir}")

    for path in pdf_files:
        print(f"[INGEST] Loading book: {path}")
        pages = load_single_book(path, llm_meta)
        chunks = chunk_documents(pages)
        print(f"[INGEST]   -> {len(pages)} pages, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"[INGEST] Total chunks across all books: {len(all_chunks)}")
    return all_chunks

def extract_book_author_from_pages(pages: list[Document]) -> str | None:
    """
    Aggressive heuristic to guess the main author from early pages.

    Strategy:
    - Scan first 5 pages.
    - Prefer lines that:
      * start with "Por " or "De " and then a name,
      * or look like "Nombre Apellido" (2–4 words, capitalized, no digits),
      * and are not clearly labeled as editor, coordinator, photographer, etc.
    """
    if not pages:
        return None

    candidate_pages = pages[:5]
    best_candidate = None

    for p in candidate_pages:
        text = p.page_content or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines:
            low = line.lower()
            # Skip obvious non-author lines
            if any(
                kw in low
                for kw in [
                    "editorial",
                    "ediciones",
                    "colección",
                    "coleccion",
                    "isbn",
                    "copyright",
                    "derechos reservados",
                    "impreso en",
                    "depósito legal",
                    "deposito legal",
                    "fotografía",
                    "fotografo",
                    "fotógrafo",
                    "coordinador",
                    "coordinadora",
                    "compilador",
                    "compiladora",
                    "prólogo de",
                    "prologo de",
                ]
            ):
                continue

            # 1) Lines like "Por Nombre Apellido"
            if low.startswith("por ") or low.startswith("de "):
                possible = line.split(maxsplit=1)[1] if len(line.split()) > 1 else ""
                if possible and not any(ch.isdigit() for ch in possible):
                    best_candidate = possible
                    break

            # 2) Generic "Nombre Apellido" style
            words = line.split()
            if 2 <= len(words) <= 4 and not any(ch.isdigit() for ch in line):
                # crude check: at least one space, mostly letters
                if all(
                    ch.isalpha() or ch.isspace() or ch in "ÁÉÍÓÚáéíóúÑñ"
                    for ch in line
                ):
                    best_candidate = line
                    break

        if best_candidate:
            break

    return best_candidate


def load_llm_metadata() -> dict[str, dict]:
    """
    Load LLM-generated metadata for each PDF from data/books_metadata_llm.json.

    Returns:
        dict[file_name, metadata_dict]
        where metadata_dict includes keys like:
        - title
        - subtitle
        - main_author
        - year
        - publisher
        - subjects
        etc.
    """
    if not LLM_METADATA_PATH.exists():
        print(f"[INGEST] No LLM metadata file found at {LLM_METADATA_PATH}, using heuristics only.")
        return {}

    try:
        data = json.loads(LLM_METADATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print(f"[INGEST] LLM metadata file has unexpected format, ignoring.")
            return {}
        return data
    except Exception as e:
        print(f"[INGEST] Failed to load LLM metadata: {e}")
        return {}

import os
import re

YEAR_PREFIX_RE = re.compile(r"^(\d{4})[_\-]")

def infer_year_from_filename(path: str) -> str | None:
    """
    Assumes filenames start with 'YYYY_' or 'YYYY-'.
    Returns the 4-digit year as a string, or None if not found.
    """
    fname = os.path.basename(path)
    m = YEAR_PREFIX_RE.match(fname)
    if m:
        return m.group(1)
    return None
