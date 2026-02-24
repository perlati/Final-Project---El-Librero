from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
BOOKS_DIR = ROOT_DIR / "data" / "books"
RENAMED_DIR = ROOT_DIR / "data" / "books_renamed"
METADATA_PATH = ROOT_DIR / "data" / "books_metadata_llm.json"


STOPWORDS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "su",
    "sus",
    "un",
    "una",
    "uno",
    "unos",
    "unas",
    "y",
}

SURNAME_PARTICLES = {
    "da",
    "das",
    "de",
    "del",
    "di",
    "la",
    "las",
    "los",
    "mc",
    "san",
    "santa",
    "van",
    "von",
    "y",
}


def _normalize_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
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


def _extract_surname(main_author: str) -> str:
    if not main_author.strip():
        return "Desconocido"

    tokens = re.findall(r"[A-Za-zÀ-ÿ'\-]+", main_author)
    if not tokens:
        return "Desconocido"

    for token in reversed(tokens):
        normalized = _normalize_ascii(token).lower()
        if normalized not in SURNAME_PARTICLES:
            return _title_case_component(token, "Desconocido")

    return _title_case_component(tokens[-1], "Desconocido")


def _short_title(title: str, fallback_from_name: str) -> str:
    base = title.strip() or fallback_from_name
    words_raw = re.findall(r"[A-Za-zÀ-ÿ0-9]+", base)
    if not words_raw:
        return "Sin_Titulo"

    normalized_words = [_normalize_ascii(word).lower() for word in words_raw]
    filtered_words = [w for w in normalized_words if w not in STOPWORDS]

    if len(filtered_words) < 3:
        filtered_words = normalized_words

    chosen = filtered_words[:6]
    joined = "_".join(chosen)
    return _title_case_component(joined, "Sin_Titulo")


def _detect_format(file_name: str) -> str:
    name = _normalize_ascii(file_name).lower()

    if "cover" in name or "cubierta" in name:
        return "COVER"
    if "ebook" in name or "epub" in name or "kdp" in name:
        return "EBOOK"
    if "proof" in name or "galera" in name:
        return "PROOF"

    return "PRINT"


def _detect_territory(country: str, publication_place: str) -> str:
    haystack = _normalize_ascii(f"{country} {publication_place}").lower()

    if "spain" in haystack or "espana" in haystack:
        return "ES"
    if "mexico" in haystack:
        return "MX"
    if "united states" in haystack or "usa" in haystack or "eeuu" in haystack:
        return "US"
    if "latam" in haystack or "latino" in haystack:
        return "LATAM"
    if "internacional" in haystack or "international" in haystack:
        return "INT"

    return "VE"


def _normalize_language(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang.startswith("en"):
        return "EN"
    if lang.startswith("pt"):
        return "PT"
    return "ES"


def _isbn13_checksum(first12: str) -> str:
    total = 0
    for index, digit_char in enumerate(first12):
        digit = int(digit_char)
        total += digit if index % 2 == 0 else digit * 3
    check = (10 - (total % 10)) % 10
    return str(check)


def _isbn10_to_isbn13(isbn10: str) -> str:
    first12 = "978" + isbn10[:9]
    return first12 + _isbn13_checksum(first12)


def _pseudo_isbn13(seed: str) -> str:
    hash_int = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    first12 = "979" + f"{hash_int % 1_000_000_000:09d}"
    return first12 + _isbn13_checksum(first12)


def _normalize_isbn(raw_isbn: str, seed: str) -> str:
    digits = re.sub(r"\D", "", raw_isbn or "")

    if len(digits) == 13:
        return digits
    if len(digits) == 10:
        return _isbn10_to_isbn13(digits)

    return _pseudo_isbn13(seed)


def _default_record_from_filename(file_name: str) -> dict[str, Any]:
    stem = Path(file_name).stem
    year_match = re.match(r"^(\d{4})", stem)
    year = year_match.group(1) if year_match else ""

    core = re.sub(r"^\d{4}_?", "", stem)
    core = _normalize_ascii(core)
    core = re.sub(
        r"(?i)\b(paper\s*back|paper_back|pb|final|tripa|kdp|print|texto|def)\b",
        " ",
        core,
    )
    core = re.sub(r"[_\-]+", " ", core)
    core = re.sub(r"\s+", " ", core).strip()

    title = core[:120] if core else Path(file_name).stem

    return {
        "title": title,
        "subtitle": "",
        "main_author": "",
        "other_authors": [],
        "editors": [],
        "translators": [],
        "photographers": [],
        "illustrators": [],
        "year": year,
        "publisher": "",
        "publication_place": "",
        "country": "",
        "isbn": "",
        "language": "es",
        "page_count": "",
        "subjects": [],
        "document_type": "",
    }


def main() -> None:
    if not BOOKS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de libros: {BOOKS_DIR}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No existe metadata: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata: dict[str, dict[str, Any]] = json.load(file)

    normalized_key_map = {_normalize_lookup(key): key for key in metadata.keys()}

    RENAMED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(path for path in BOOKS_DIR.iterdir() if path.suffix.lower() == ".pdf")
    used_names: set[str] = set()
    added_metadata_entries = 0
    copied_files = 0
    missing_metadata = []
    generated_metadata_entries = 0

    for pdf_path in pdf_files:
        original_name = pdf_path.name
        record = metadata.get(original_name)

        if record is None:
            lookup = _normalize_lookup(original_name)
            matched_key = normalized_key_map.get(lookup)
            if matched_key:
                record = metadata[matched_key]

        if record is None:
            record = _default_record_from_filename(original_name)
            metadata[original_name] = dict(record)
            generated_metadata_entries += 1
            missing_metadata.append(original_name)

        title = str(record.get("title", ""))
        main_author = str(record.get("main_author", ""))
        editors = record.get("editors", [])
        country = str(record.get("country", ""))
        publication_place = str(record.get("publication_place", ""))
        language = str(record.get("language", ""))
        raw_isbn = str(record.get("isbn", ""))
        year = str(record.get("year", "")).strip()

        identifier = _normalize_isbn(raw_isbn, seed=original_name)
        effective_author = main_author
        if not effective_author.strip() and isinstance(editors, list) and editors:
            effective_author = str(editors[0])
        surname = _extract_surname(effective_author)

        fallback_title = re.sub(r"^\d{4}_", "", pdf_path.stem)
        short_title = _short_title(title, fallback_from_name=fallback_title)

        file_format = _detect_format(original_name)
        territory = _detect_territory(country, publication_place)
        lang = _normalize_language(language)
        
        # Usar año de publicación o fecha actual como fallback
        year_or_date = year if year and len(year) == 4 else datetime.now().strftime("%Y%m%d")

        base = f"{identifier}_{surname}_{short_title}_{file_format}_{territory}_{lang}"
        version = 1
        while True:
            candidate_name = f"{base}_v{version:02d}_{year_or_date}.pdf"
            if candidate_name not in used_names:
                used_names.add(candidate_name)
                break
            version += 1

        destination = RENAMED_DIR / candidate_name
        shutil.copy2(pdf_path, destination)
        copied_files += 1

        if candidate_name not in metadata:
            metadata[candidate_name] = dict(record)
            added_metadata_entries += 1

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"PDFs procesados: {len(pdf_files)}")
    print(f"PDFs copiados a '{RENAMED_DIR}': {copied_files}")
    print(f"Nuevas claves agregadas en metadata: {added_metadata_entries}")
    print(f"Entradas metadata autogeneradas: {generated_metadata_entries}")
    print(f"PDFs sin metadata: {len(missing_metadata)}")

    if missing_metadata:
        print("Archivos sin metadata:")
        for item in missing_metadata:
            print(f"- {item}")


if __name__ == "__main__":
    main()
