# backend/ingestion/extract_metadata_llm.py

import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

from backend.config import BOOKS_DIR, LLM_MODEL
from backend.ingestion.pdf_loader import list_pdf_files, infer_book_title_and_year_from_filename

OUTPUT_PATH = Path("data/books_metadata_llm.json")


def build_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def make_prompt(text: str) -> str:
    return f"""
You are a bibliographic metadata expert for a Latin American publishing house.

You will receive the beginning of a book (cover, legal page, index, first pages) as plain text.
Your task is to extract a clean, standard metadata record.

Return ONLY a valid JSON object with this exact schema, nothing else:

{{
  "title": "",
  "subtitle": "",
  "main_author": "",
  "other_authors": [],
  "editors": [],
  "translators": [],
  "photographers": [],
  "illustrators": [],
  "year": "",
  "publisher": "",
  "publication_place": "",
  "country": "",
  "isbn": "",
  "language": "",
  "page_count": "",
  "subjects": [],
  "document_type": ""
}}

Rules (VERY IMPORTANT):

1. TITLE AND SUBTITLE
   - "title": the main title of the work as it should appear in the catalog.
     Do NOT include series names, collection names or technical labels.
     Do NOT include words like "Paper Back", "Tripa", "Final", "Edición", "Colección".
   - "subtitle": only the subtitle, if clearly indicated; otherwise "".

2. MAIN AUTHOR vs OTHER NAMES
   - "main_author": the primary literary author of the book (narrative, essay, history, etc.).
     This is the name used to cite the book in an academic bibliography.
   - DO NOT put editors, coordinators, prologue writers, photographers or illustrators as main_author.
   - "other_authors": other literary authors if there are several (e.g., co-authors of the text).
   - If the only names you see are clearly labeled as "editor", "coordinador", "compilador",
     "fotógrafo", "fotografo", "fotografía", "ilustrador", "prólogo de", etc.,
     then leave "main_author" = "" and put those names in the corresponding roles (editors, photographers, illustrators).

3. ROLES
   - "editors": list of names explicitly labeled as editor / coordinator / compilador.
   - "translators": names labeled as translator / traductor.
   - "photographers": names labeled as photographer / fotógrafo / fotografia.
   - "illustrators": names labeled as illustrator / ilustrador.

4. YEAR
   - "year": 4-digit publication year as a string.
   - Look for dates on the legal/credits page (e.g. "Primera edición 2018", "© Editorial X, 2012").
   - If you are not sure, but see several possible years, choose the one that most likely corresponds to the book's publication.
   - If you really cannot infer a publication year, set "year" = "".

5. OTHER FIELDS
   - "publisher": name of the publishing house.
   - "publication_place": city of publication if visible.
   - "country": country if visible.
   - "isbn": ISBN if visible, else "".
   - "language": "es" for Spanish, "en" for English, etc. Infer from the text.
   - "page_count": total pages if visible, else "".
   - "subjects": list of 2–6 short subject labels (e.g., ["política venezolana", "democracia", "historia contemporánea"]).
   - "document_type": short label like "ensayo", "historia", "crónica", "novela", "biografía", "memorias", etc.

Formatting:
- All person names should be in "Nombre Apellido" format with correct capitalization.
- Return ONLY the JSON object. No commentary, no explanation.

Book text:
--------------------
{text}
--------------------
"""


def extract_metadata_for_pdf(path: str, llm: ChatOpenAI) -> dict:
    loader = PyPDFLoader(path)
    pages = loader.load()

    # Extract year from filename as fallback
    _, inferred_year = infer_book_title_and_year_from_filename(path)

    # First N pages as context for metadata
    N = 5
    raw_text = "\n\n".join(p.page_content or "" for p in pages[:N])
    raw_text = raw_text[:6000]  # safety truncation

    prompt = make_prompt(raw_text)
    resp = llm.invoke(prompt)
    content = resp.content.strip()

    try:
        data = json.loads(content)
    except Exception:
        data = {}

    # Normalize and ensure all keys exist
    def get_str(key):
        return str(data.get(key, "") or "").strip()

    def get_list(key):
        val = data.get(key, [])
        if not isinstance(val, list):
            return []
        return [str(x).strip() for x in val if str(x).strip()]

    # Use LLM year if available, otherwise fallback to filename year
    llm_year = get_str("year")
    final_year = llm_year if llm_year else (inferred_year or "")

    record = {
        "title": get_str("title"),
        "subtitle": get_str("subtitle"),
        "main_author": get_str("main_author"),
        "other_authors": get_list("other_authors"),
        "editors": get_list("editors"),
        "translators": get_list("translators"),
        "photographers": get_list("photographers"),
        "illustrators": get_list("illustrators"),
        "year": final_year,
        "publisher": get_str("publisher"),
        "publication_place": get_str("publication_place"),
        "country": get_str("country"),
        "isbn": get_str("isbn"),
        "language": get_str("language"),
        "page_count": get_str("page_count"),
        "subjects": get_list("subjects"),
        "document_type": get_str("document_type"),
    }

    return record


def main():
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    llm = build_llm()

    pdf_files = list_pdf_files(BOOKS_DIR)
    print(f"[META] Found {len(pdf_files)} PDF files under {BOOKS_DIR}")

    results = {}

    for path in pdf_files:
        file_name = os.path.basename(path)
        print(f"[META] Extracting metadata for: {file_name}")
        meta = extract_metadata_for_pdf(path, llm)
        results[file_name] = meta
        print(f"       -> {meta}")

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[META] Saved metadata to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
