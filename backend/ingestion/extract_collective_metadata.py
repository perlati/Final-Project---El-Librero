#!/usr/bin/env python3
"""Extrae metadata editorial de libros colectivos directamente desde PDFs (sin API externa)."""

import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


PROJECT_ROOT = Path(__file__).parent.parent.parent
BOOKS_DIR = PROJECT_ROOT / "data" / "books"
METADATA_FILE = PROJECT_ROOT / "data" / "books_metadata_llm.json"


def _clean_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,:;|-\t\n\r")
    return value


def _split_names(value: str) -> list[str]:
    value = value.replace(" y ", "|").replace(",", "|")
    candidates = [_clean_name(part) for part in value.split("|")]
    return [name for name in candidates if len(name) > 2]


def _extract_first_pages_text(pdf_path: Path, limit_pages: int = 12) -> list[str]:
    pages = PyPDFLoader(str(pdf_path)).load()
    return [p.page_content for p in pages[: min(limit_pages, len(pages))]]


def _extract_isbn(text: str) -> str:
    match = re.search(r"ISBN\s*[:\-]?\s*([0-9\-\s]{10,20})", text, flags=re.IGNORECASE)
    if not match:
        return ""
    digits = re.sub(r"\D", "", match.group(1))
    return digits if len(digits) in (10, 13) else ""


def _extract_year(text: str) -> str:
    years = re.findall(r"(?:19|20)\d{2}", text)
    if not years:
        return ""
    return min(years)


def _extract_publisher(text: str) -> str:
    if re.search(r"Editorial\s+Dahbar", text, flags=re.IGNORECASE):
        return "Editorial Dahbar"
    if re.search(r"Editorial\s+Melvin", text, flags=re.IGNORECASE):
        return "Editorial Melvin"
    if re.search(r"Fundaci[oó]n\s+Educativa\s+San\s+Judas\s+Tadeo", text, flags=re.IGNORECASE):
        return "Fundación Educativa San Judas Tadeo"
    return ""


def _extract_editors(text: str) -> list[str]:
    editors: list[str] = []

    m = re.search(r"©\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\s]+?)\s*\((?:compilador|coordinador|editor)\)", text, flags=re.IGNORECASE)
    if m:
        editors.extend(_split_names(m.group(1)))

    m = re.search(r"Coord\.\s*([^\n|]+?)\s+y\s+([^\n|]+)", text)
    if m:
        editors.extend([_clean_name(m.group(1)), _clean_name(m.group(2))])

    m = re.search(r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\s]+?)\s+y\s+([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\s]+)\s*\|\s*Editores", text)
    if m:
        editors.extend([_clean_name(m.group(1)), _clean_name(m.group(2))])

    m = re.search(r"\|\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\s]+)\s*\|\s*Editor", text)
    if m:
        editors.extend(_split_names(m.group(1)))

    dedup: list[str] = []
    for name in editors:
        if name and name not in dedup:
            dedup.append(name)
    return dedup


def _extract_corporate_author(text: str) -> str:
    if re.search(r"IPYS\s+Venezuela\s+es\s+una\s+ONG", text, flags=re.IGNORECASE):
        return "IPYS Venezuela"
    return ""


def main() -> None:
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    targets = [
        key
        for key, value in metadata.items()
        if not key.startswith("97") and (not value.get("main_author") or value.get("main_author") == "Desconocido")
    ]

    print(f"Procesando {len(targets)} libros sin autor principal...")

    updated = 0
    for filename in targets:
        pdf_path = BOOKS_DIR / filename
        if not pdf_path.exists():
            continue

        try:
            pages_text = _extract_first_pages_text(pdf_path, limit_pages=12)
            merged_text = "\n".join(pages_text)
        except Exception as error:
            print(f"❌ {filename}: no se pudo leer PDF ({error})")
            continue

        corporate_author = _extract_corporate_author(merged_text)
        editors = _extract_editors(merged_text)
        isbn = _extract_isbn(merged_text)
        year = _extract_year(merged_text)
        publisher = _extract_publisher(merged_text)

        record = metadata[filename]
        if corporate_author and not record.get("main_author"):
            record["main_author"] = corporate_author
        if editors:
            record["editors"] = editors
        if isbn and not record.get("isbn"):
            record["isbn"] = isbn
        if year and not record.get("year"):
            record["year"] = year
        if publisher and not record.get("publisher"):
            record["publisher"] = publisher

        metadata[filename] = record
        updated += 1
        print(f"✅ {filename}: author={record.get('main_author','')}, editors={record.get('editors', [])}")

    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nProceso completado. Registros revisados: {updated}")


if __name__ == "__main__":
    main()
