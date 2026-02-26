from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.config import PROJECT_ROOT
from backend.utils.text import normalize_title

MANIFEST_PATH = Path(PROJECT_ROOT) / "data" / "website_media_curated" / "curation_manifest.json"
METADATA_PATH = Path(PROJECT_ROOT) / "data" / "books_metadata_llm.json"
OUTPUT_ROOT = Path(PROJECT_ROOT) / "data" / "website_media_organized"
INDEX_PATH = OUTPUT_ROOT / "asset_index.json"

AUTHOR_HINTS = {
    "autor", "author", "authors", "foto", "fotos", "portrait", "headshot", "perfil", "retrato", "semblanza", "biografia",
}

STOPWORDS = {
    "de", "del", "la", "las", "el", "los", "y", "en", "a", "un", "una", "por", "para", "con", "sin",
}


@dataclass
class BookRef:
    title: str
    title_slug: str
    title_tokens: set[str]
    author: str
    author_slug: str
    author_tokens: set[str]
    year: str
    isbn: str
    cover_basename: str


@dataclass
class AuthorRef:
    name: str
    slug: str
    tokens: set[str]


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    value = value.replace("ü", "u").replace("ñ", "n")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "unknown"


def _tokenize(value: str) -> List[str]:
    tokens = re.findall(r"[a-záéíóúüñ0-9]+", value.lower())
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


def _base_name(stem: str) -> str:
    key = stem.lower().strip()
    key = re.sub(r"[-_](\d{2,5})x(\d{2,5})$", "", key)
    key = re.sub(r"[-_](scaled|copy|copia|final|edited|editado|small|medium|large)$", "", key)
    key = re.sub(r"[-_](\d{1,2})$", "", key)
    key = re.sub(r"\s+", " ", key)
    return key.strip(" -_")


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_books_and_authors() -> Tuple[List[BookRef], List[AuthorRef]]:
    if not METADATA_PATH.exists():
        return [], []

    raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict):
        items = [item for item in raw.values() if isinstance(item, dict)]

    books: List[BookRef] = []
    authors_map: Dict[str, AuthorRef] = {}

    for item in items:
        web_catalog = item.get("web_catalog") if isinstance(item.get("web_catalog"), dict) else {}

        title = normalize_title(item.get("normalized_title") or item.get("title", "")).strip()
        if not title:
            continue

        author = (
            (item.get("author") or "").strip()
            or (item.get("main_author") or "").strip()
            or (web_catalog.get("web_author") or "").strip()
        )

        year = str(item.get("year") or item.get("pub_year") or "").strip()
        isbn = str(item.get("isbn") or "").strip()

        cover_url = str(web_catalog.get("web_cover_url") or "").strip()
        cover_basename = ""
        if cover_url:
            cover_basename = _base_name(Path(cover_url).stem)

        title_tokens = set(_tokenize(title))
        author_tokens = set(_tokenize(author)) if author else set()

        book = BookRef(
            title=title,
            title_slug=_slug(title),
            title_tokens=title_tokens,
            author=author,
            author_slug=_slug(author) if author else "author-unknown",
            author_tokens=author_tokens,
            year=year,
            isbn=isbn,
            cover_basename=cover_basename,
        )
        books.append(book)

        if author and author not in authors_map:
            authors_map[author] = AuthorRef(name=author, slug=_slug(author), tokens=author_tokens)

    return books, list(authors_map.values())


def _best_book_match(file_stem: str, rel_path: str, books: List[BookRef]) -> Tuple[BookRef | None, float, str]:
    stem_base = _base_name(file_stem)
    stem_tokens = set(_tokenize(stem_base))
    rel_tokens = set(_tokenize(rel_path))
    combined = stem_tokens | rel_tokens

    best: BookRef | None = None
    best_score = 0.0
    reason = ""

    for book in books:
        if not book.title_tokens:
            continue

        overlap = len(combined & book.title_tokens)
        overlap_ratio = overlap / max(1, len(book.title_tokens))
        score = overlap * 2.0 + overlap_ratio

        if book.cover_basename and book.cover_basename == stem_base:
            score += 4.5

        if book.cover_basename and book.cover_basename in stem_base:
            score += 2.0

        if book.isbn and book.isbn in rel_path:
            score += 5.0

        if book.author_tokens and len(combined & book.author_tokens) >= 1:
            score += 0.6

        if score > best_score:
            best_score = score
            best = book
            reason = f"title_overlap={overlap}, ratio={overlap_ratio:.2f}"

    if best_score >= 3.2:
        return best, best_score, reason
    return None, best_score, reason


def _best_author_match(file_stem: str, rel_path: str, authors: List[AuthorRef]) -> Tuple[AuthorRef | None, float]:
    text_tokens = set(_tokenize(file_stem)) | set(_tokenize(rel_path))
    best: AuthorRef | None = None
    best_score = 0.0

    for author in authors:
        if not author.tokens:
            continue
        overlap = len(text_tokens & author.tokens)
        if overlap == 0:
            continue
        ratio = overlap / max(1, len(author.tokens))
        score = overlap * 1.6 + ratio
        if score > best_score:
            best_score = score
            best = author

    return best, best_score


def _looks_like_author_photo(file_stem: str, rel_path: str) -> bool:
    text_tokens = set(_tokenize(file_stem)) | set(_tokenize(rel_path))
    return any(tok in AUTHOR_HINTS for tok in text_tokens)


def _target_dir_and_name(
    cluster_id: int,
    source_ext: str,
    source_stem: str,
    year: str,
    final_type: str,
    book: BookRef | None,
    author: AuthorRef | None,
) -> Tuple[Path, str]:
    source_slug = _slug(_base_name(source_stem))

    if final_type == "cover" and book is not None:
        out_dir = OUTPUT_ROOT / "covers" / "by_book" / book.title_slug
        name = f"cover__{book.year or year or 'unk'}__{book.title_slug}__{book.author_slug}__c{cluster_id:05d}{source_ext}"
        return out_dir, name

    if final_type == "author_photo":
        author_slug = author.slug if author else "author-unknown"
        out_dir = OUTPUT_ROOT / "author_photos" / "by_author" / author_slug
        name = f"author__{author_slug}__{year or 'unk'}__c{cluster_id:05d}{source_ext}"
        return out_dir, name

    if final_type == "branding":
        out_dir = OUTPUT_ROOT / "branding"
        name = f"branding__{source_slug}__{year or 'unk'}__c{cluster_id:05d}{source_ext}"
        return out_dir, name

    if final_type == "web_media":
        out_dir = OUTPUT_ROOT / "web_media"
        name = f"media__{source_slug}__{year or 'unk'}__c{cluster_id:05d}{source_ext}"
        return out_dir, name

    out_dir = OUTPUT_ROOT / "uncertain"
    name = f"uncertain__{source_slug}__{year or 'unk'}__c{cluster_id:05d}{source_ext}"
    return out_dir, name


def organize() -> Dict[str, Any]:
    manifest = _load_manifest()
    books, authors = _load_books_and_authors()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    clusters = manifest.get("clusters", [])
    index_entries: List[Dict[str, Any]] = []

    counts = {
        "cover": 0,
        "author_photo": 0,
        "branding": 0,
        "web_media": 0,
        "uncertain": 0,
    }

    source_root = Path(PROJECT_ROOT) / manifest.get("source_root", "data/website_media/MEDIA")

    for cluster in clusters:
        canonical = cluster.get("canonical", {})
        if not canonical:
            continue

        cluster_id = int(cluster.get("cluster_id") or 0)
        rel_path = str(canonical.get("rel_path") or "")
        curated_path = str(canonical.get("curated_path") or "")
        if not rel_path:
            continue

        src_path = Path(PROJECT_ROOT) / curated_path if curated_path else source_root / rel_path
        if not src_path.exists():
            src_path = source_root / rel_path
            if not src_path.exists():
                continue

        stem = Path(rel_path).stem
        ext = src_path.suffix.lower()
        year = str(canonical.get("year") or "")
        current_category = str(cluster.get("category") or "web_media")

        book_match, book_score, book_reason = _best_book_match(stem, rel_path, books)
        author_match, author_score = _best_author_match(stem, rel_path, authors)

        final_type = "web_media"
        reason = "default"

        if current_category == "branding":
            final_type = "branding"
            reason = "from_branding_cluster"
        elif _looks_like_author_photo(stem, rel_path) and (author_score >= 2.0 or (book_match is None and author_score >= 1.3)):
            final_type = "author_photo"
            reason = "author_hint"
        elif current_category == "cover":
            if book_match is not None:
                final_type = "cover"
                reason = "cover_cluster+book_match"
            elif author_score >= 1.3:
                final_type = "author_photo"
                reason = "cover_cluster_but_author_like"
            else:
                final_type = "web_media"
                reason = "cover_cluster_without_book_match"
        else:
            if book_match is not None and book_score >= 4.2:
                final_type = "cover"
                reason = "web_media_promoted_to_cover"
            elif author_score >= 2.0:
                final_type = "author_photo"
                reason = "author_token_match"
            else:
                final_type = "web_media"
                reason = "kept_web_media"

        out_dir, out_name = _target_dir_and_name(
            cluster_id=cluster_id,
            source_ext=ext,
            source_stem=stem,
            year=year,
            final_type=final_type,
            book=book_match,
            author=author_match,
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / out_name
        if not out_path.exists():
            shutil.copy2(src_path, out_path)

        counts[final_type] += 1

        index_entries.append(
            {
                "cluster_id": cluster_id,
                "source_rel_path": rel_path,
                "source_curated_path": curated_path,
                "final_type": final_type,
                "final_path": out_path.relative_to(PROJECT_ROOT).as_posix(),
                "decision_reason": reason,
                "book_link": {
                    "title": book_match.title if book_match else "",
                    "title_slug": book_match.title_slug if book_match else "",
                    "author": book_match.author if book_match else "",
                    "year": book_match.year if book_match else "",
                    "score": round(book_score, 3),
                    "match_reason": book_reason,
                },
                "author_link": {
                    "name": author_match.name if author_match else "",
                    "slug": author_match.slug if author_match else "",
                    "score": round(author_score, 3),
                },
            }
        )

    index_payload = {
        "source_manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "output_root": OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "total_assets_processed": len(index_entries),
        "counts": counts,
        "entries": index_entries,
    }

    INDEX_PATH.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_payload


def main() -> None:
    payload = organize()
    print("[DONE] Visual assets organized")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Index: {INDEX_PATH}")
    print(f"Processed: {payload['total_assets_processed']}")
    print(f"Counts: {payload['counts']}")


if __name__ == "__main__":
    main()
