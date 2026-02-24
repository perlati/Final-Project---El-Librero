#!/usr/bin/env python3

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "wc-product-export-24-2-2026-1771940340217.csv"
METADATA_PATH = DATA_DIR / "books_metadata_llm.json"
BOOKS_DIR = DATA_DIR / "books"
COVERS_DIR = DATA_DIR / "covers_renamed"
REPORTS_DIR = DATA_DIR / "reconciliation_reports"


STOPWORDS = {
    "a", "al", "con", "de", "del", "el", "en", "es", "la", "las", "lo", "los",
    "para", "por", "que", "se", "sin", "su", "sus", "un", "una", "uno", "unos", "unas", "y",
}

SURNAME_PARTICLES = {
    "da", "das", "de", "del", "di", "la", "las", "los", "mc", "san", "santa", "van", "von", "y",
}


def _normalize_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return normalized.encode("ascii", "ignore").decode("ascii")


def _normalize_lookup(text: str) -> str:
    text = _normalize_ascii(text).lower()
    text = text.replace("_", "").replace("-", "").replace(" ", "")
    return re.sub(r"[^a-z0-9.]", "", text)


def _sanitize_component(text: str, default: str) -> str:
    clean = _normalize_ascii(text)
    clean = re.sub(r"[^A-Za-z0-9]+", "_", clean)
    clean = clean.strip("_")
    return clean or default


def _title_case_component(text: str, default: str) -> str:
    clean = _sanitize_component(text, default)
    return "_".join(part[:1].upper() + part[1:] for part in clean.split("_") if part)


def _extract_surname(person_or_author: str) -> str:
    text = (person_or_author or "").strip()
    if not text:
        return "Desconocido"

    first_person = re.split(r",|\sy\s|\|", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    tokens = re.findall(r"[A-Za-zÀ-ÿ'\-]+", first_person)
    if not tokens:
        return "Desconocido"

    for token in reversed(tokens):
        normalized = _normalize_ascii(token).lower()
        if normalized not in SURNAME_PARTICLES:
            return _title_case_component(token, "Desconocido")

    return _title_case_component(tokens[-1], "Desconocido")


def _short_title(title: str) -> str:
    words_raw = re.findall(r"[A-Za-zÀ-ÿ0-9]+", title or "")
    if not words_raw:
        return "Sin_Titulo"

    normalized_words = [_normalize_ascii(word).lower() for word in words_raw]
    filtered_words = [word for word in normalized_words if word not in STOPWORDS]
    chosen = (filtered_words or normalized_words)[:6]
    return _title_case_component("_".join(chosen), "Sin_Titulo")


def _normalize_isbn(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 13:
        return digits
    if len(digits) == 10:
        return "978" + digits[:9]
    return ""


def _first_image_url(images_field: str) -> str:
    if not images_field:
        return ""
    first = images_field.split(",")[0].strip()
    if first.startswith("http://") or first.startswith("https://"):
        return first
    return ""


def _extract_year(web_row: dict[str, str]) -> str:
    for key in ("Meta: edicion", "Nombre"):
        value = web_row.get(key, "")
        match = re.search(r"(19|20)\d{2}", value or "")
        if match:
            return match.group(0)
    return ""


def _build_web_payload(row: dict[str, str]) -> dict[str, Any]:
    return {
        "web_id": row.get("ID", ""),
        "web_sku": row.get("SKU", ""),
        "web_name": row.get("Nombre", ""),
        "web_published": row.get("Publicado", ""),
        "web_visibility": row.get("Visibilidad en el catálogo", ""),
        "web_short_description": row.get("Descripción corta", ""),
        "web_description": row.get("Descripción", ""),
        "web_categories": row.get("Categorías", ""),
        "web_tags": row.get("Etiquetas", ""),
        "web_brand": row.get("Marcas", ""),
        "web_publisher": row.get("Meta: sello_editorial", ""),
        "web_edition": row.get("Meta: edicion", ""),
        "web_format": row.get("Meta: formato", ""),
        "web_code": row.get("Meta: codigo", ""),
        "web_author": row.get("Meta: autor", ""),
        "web_page_count": row.get("Meta: numeropagina", ""),
        "web_cover_url": _first_image_url(row.get("Imágenes", "")),
    }


def _merge_record(record: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    isbn = _normalize_isbn(row.get("Meta: isbn", ""))
    year = _extract_year(row)
    web_author = (row.get("Meta: autor", "") or "").strip()
    web_publisher = (row.get("Meta: sello_editorial", "") or "").strip()
    web_pages = (row.get("Meta: numeropagina", "") or "").strip()

    updated = dict(record)

    title = (updated.get("title") or "").strip()
    if not title:
        updated["title"] = (row.get("Nombre", "") or "").strip()

    current_author = (updated.get("main_author") or "").strip()
    if (not current_author or current_author.lower() == "desconocido") and web_author and web_author.lower() not in {"varios", "autor", "autores"}:
        updated["main_author"] = web_author

    if not (updated.get("isbn") or "").strip() and isbn:
        updated["isbn"] = isbn

    if not (updated.get("year") or "").strip() and year:
        updated["year"] = year

    if not (updated.get("publisher") or "").strip() and web_publisher:
        updated["publisher"] = web_publisher

    if not (updated.get("page_count") or "").strip() and web_pages:
        updated["page_count"] = web_pages

    # Datos web extendidos
    updated["web_catalog"] = _build_web_payload(row)

    return updated


def _find_best_match(row: dict[str, str], metadata: dict[str, dict[str, Any]], original_keys: list[str], isbn_map: dict[str, str], title_map: dict[str, str]) -> str | None:
    row_isbn = _normalize_isbn(row.get("Meta: isbn", ""))
    if row_isbn and row_isbn in isbn_map:
        return isbn_map[row_isbn]

    row_name = (row.get("Nombre", "") or "").strip()
    if row_name:
        key = _normalize_lookup(row_name)
        if key in title_map:
            return title_map[key]

    return None


def _looks_like_image(content: bytes, content_type: str) -> bool:
    if not content:
        return False

    ctype = (content_type or "").lower()
    if ctype.startswith("image/"):
        return True

    magic = content[:16]
    if magic.startswith(b"\xff\xd8\xff"):
        return True  # jpg
    if magic.startswith(b"\x89PNG\r\n\x1a\n"):
        return True  # png
    if magic.startswith(b"RIFF") and b"WEBP" in content[:32]:
        return True  # webp

    return False


def _download_cover(url: str, output_path: Path) -> bool:
    if not url:
        return False
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://editorialdahbar.com/",
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        if not _looks_like_image(response.content, response.headers.get("Content-Type", "")):
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return True
    except Exception:
        return False


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV no encontrado: {CSV_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata no encontrada: {METADATA_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    metadata: dict[str, dict[str, Any]] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    original_keys = [key for key in metadata if not key.startswith("97")]

    isbn_map: dict[str, str] = {}
    title_map: dict[str, str] = {}
    for key in original_keys:
        record = metadata[key]
        record_isbn = _normalize_isbn(str(record.get("isbn", "")))
        if record_isbn:
            isbn_map[record_isbn] = key
        title = (record.get("title") or "").strip()
        if title:
            title_map[_normalize_lookup(title)] = key

    web_rows = [
        row for row in rows
        if (row.get("Tipo", "").strip().lower() in {"simple", "variable", "grouped"}) and (row.get("Nombre", "").strip())
    ]

    matched = 0
    unmatched_web: list[dict[str, Any]] = []
    matched_keys: set[str] = set()

    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_covers = 0
    used_cover_names: set[str] = set(path.name for path in COVERS_DIR.glob("*"))
    cover_failures: list[dict[str, str]] = []

    for row in web_rows:
        matched_key = _find_best_match(row, metadata, original_keys, isbn_map, title_map)
        if not matched_key:
            unmatched_web.append({
                "web_id": row.get("ID", ""),
                "name": row.get("Nombre", ""),
                "isbn": _normalize_isbn(row.get("Meta: isbn", "")),
                "sku": row.get("SKU", ""),
                "cover_url": _first_image_url(row.get("Imágenes", "")),
            })
            continue

        matched += 1
        matched_keys.add(matched_key)
        metadata[matched_key] = _merge_record(metadata[matched_key], row)

    # Descargar tapa para TODOS los libros del CSV (aunque no estén localmente)
    for row in web_rows:
        matched_key = _find_best_match(row, metadata, original_keys, isbn_map, title_map)
        record = metadata[matched_key] if matched_key else {}

        isbn = _normalize_isbn(str(record.get("isbn", ""))) or _normalize_isbn(row.get("Meta: isbn", "")) or "0000000000000"
        author = (record.get("main_author") or "").strip() or (row.get("Meta: autor", "") or "").strip()
        editors = record.get("editors") or []
        if not author and isinstance(editors, list) and editors:
            author = str(editors[0])
        if not author:
            author = "Desconocido"

        surname = _extract_surname(author)
        title = (record.get("title") or row.get("Nombre") or "Sin titulo").strip()
        short_title = _short_title(title)
        year = (record.get("year") or _extract_year(row) or "0000").strip()
        year = year if re.fullmatch(r"\d{4}", year) else "0000"

        cover_url = _first_image_url(row.get("Imágenes", ""))
        if not cover_url:
            continue

        parsed = urlparse(cover_url)
        ext = Path(parsed.path).suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"

        base = f"{isbn}_{surname}_{short_title}_COVER_VE_ES"
        version = 1
        while True:
            cover_name = f"{base}_v{version:02d}_{year}{ext}"
            if cover_name not in used_cover_names:
                used_cover_names.add(cover_name)
                break
            version += 1

        output_path = COVERS_DIR / cover_name
        if _download_cover(cover_url, output_path):
            downloaded_covers += 1
        else:
            cover_failures.append({"name": row.get("Nombre", ""), "url": cover_url, "target": cover_name})

    # libros locales sin match en web
    local_without_web = []
    for key in original_keys:
        if key not in matched_keys:
            rec = metadata[key]
            local_without_web.append({
                "file": key,
                "title": rec.get("title", ""),
                "isbn": _normalize_isbn(str(rec.get("isbn", ""))),
                "main_author": rec.get("main_author", ""),
            })

    # sincronizar entradas renombradas por isbn/título
    for key, record in list(metadata.items()):
        if not key.startswith("97"):
            continue
        isbn = _normalize_isbn(str(record.get("isbn", "")))
        title = _normalize_lookup(str(record.get("title", "")))

        source_key = None
        if isbn and isbn in isbn_map:
            source_key = isbn_map[isbn]
        elif title and title in title_map:
            source_key = title_map[title]

        if source_key and source_key in metadata:
            source = metadata[source_key]
            merged = dict(record)
            for field in ["title", "subtitle", "main_author", "other_authors", "editors", "year", "publisher", "isbn", "page_count", "subjects", "web_catalog"]:
                if source.get(field) not in (None, "", [], {}):
                    merged[field] = source.get(field)
            metadata[key] = merged

    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (REPORTS_DIR / "web_books_missing_in_local.json").write_text(
        json.dumps(unmatched_web, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORTS_DIR / "local_books_without_web_match.json").write_text(
        json.dumps(local_without_web, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORTS_DIR / "cover_download_failures.json").write_text(
        json.dumps(cover_failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Filas web procesadas: {len(web_rows)}")
    print(f"Libros reconciliados con metadata local: {matched}")
    print(f"Libros en web que faltan en local: {len(unmatched_web)}")
    print(f"Libros locales sin match web: {len(local_without_web)}")
    print(f"Tapas descargadas en {COVERS_DIR}: {downloaded_covers}")
    print(f"Fallas al descargar tapas: {len(cover_failures)}")


if __name__ == "__main__":
    main()
