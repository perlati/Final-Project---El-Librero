from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI

from backend.config import PROJECT_ROOT, LLM_MODEL
from backend.utils.text import normalize_title

METADATA_PATH = Path(PROJECT_ROOT) / "data" / "books_metadata_llm.json"
OUTPUT_PATH = Path(PROJECT_ROOT) / "data" / "books_press_critique.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# Avoid noisy LangSmith 429s during large local batch jobs.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


@dataclass
class PressSource:
    source_type: str
    title: str
    url: str
    domain: str
    snippet: str


def _load_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_PATH.exists():
        return {}
    raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def _dedupe_books_by_title(metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_title: Dict[str, Dict[str, Any]] = {}
    for key, value in metadata.items():
        title = normalize_title(value.get("normalized_title") or value.get("title", ""))
        if not title:
            continue

        normalized = title.lower()
        existing = by_title.get(normalized)
        candidate = {
            "key": key,
            "title": title,
            "author": (value.get("main_author") or value.get("author") or "").strip(),
            "year": str(value.get("year") or value.get("pub_year") or "").strip(),
            "metadata": value,
        }

        if existing is None:
            by_title[normalized] = candidate
            continue

        # Prefer original filename keys over renamed keys for source stability.
        existing_is_renamed = existing["key"].startswith("97")
        candidate_is_renamed = candidate["key"].startswith("97")
        if existing_is_renamed and not candidate_is_renamed:
            by_title[normalized] = candidate

    return list(by_title.values())


def _html_to_text(html: str, max_chars: int = 2500) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _extract_links_from_html(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    links: List[Dict[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        text = a.get_text(" ", strip=True)
        if href.startswith("http://") or href.startswith("https://"):
            links.append({"url": href, "title": text})
    # keep order, unique
    seen = set()
    out: List[Dict[str, str]] = []
    for link in links:
        url = link["url"]
        if url not in seen:
            seen.add(url)
            out.append(link)
    return out


def _is_noisy_internal_catalog_link(url: str) -> bool:
    lower = url.lower()
    return "editorialdahbar.com/libros/" in lower


def _source_type_from_url(url: str) -> str:
    lower = url.lower()
    if any(token in lower for token in ["interview", "entrevista"]):
        return "entrevista"
    if any(token in lower for token in ["review", "resena", "reseña", "critica", "crítica"]):
        return "reseña"
    if any(token in lower for token in ["opinion", "column", "articulo", "artículo"]):
        return "opinión"
    if "editorialdahbar.com" in lower:
        return "ficha_editorial"
    return "nota_web"


def _fetch_web_text(url: str, timeout: int = 12) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return _html_to_text(response.text, max_chars=2500)
    except Exception:
        return ""


def _search_web(title: str, author: str, max_results: int = 3) -> List[str]:
    query = f'"{title}" {author} reseña crítica entrevista'
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        urls: List[str] = []
        for result in soup.select("a.result__a"):
            href = (result.get("href") or "").strip()
            if href.startswith("http://") or href.startswith("https://"):
                urls.append(href)
            if len(urls) >= max_results:
                break
        return urls
    except Exception:
        return []


def _extract_json_object(raw: str) -> Dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        candidate = json.loads(raw)
        if isinstance(candidate, dict):
            return candidate
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        candidate = json.loads(match.group(0))
        if isinstance(candidate, dict):
            return candidate
    except Exception:
        return None
    return None


def _summarize_press(book_title: str, sources: List[PressSource], llm: ChatOpenAI) -> Dict[str, Any]:
    if not sources:
        return {
            "status": "sin_fuentes",
            "summary": "No se encontraron fuentes de prensa/crítica suficientes para este libro.",
            "tone": "no_disponible",
            "key_points": [],
        }

    source_block = "\n\n".join(
        [
            f"[{idx + 1}] Tipo: {src.source_type} | Dominio: {src.domain} | URL: {src.url}\n"
            f"Extracto: {src.snippet}"
            for idx, src in enumerate(sources)
        ]
    )

    prompt = (
        "Eres analista editorial de prensa y crítica literaria.\n"
        "Resume qué se dijo del libro usando SOLO las fuentes proporcionadas.\n"
        "Si las fuentes son limitadas o ambiguas, dilo explícitamente.\n"
        "Devuelve JSON válido con esta estructura exacta:\n"
        "{\n"
        '  "status": "ok|fuentes_limitadas|sin_fuentes",\n'
        '  "summary": "resumen en 4-6 frases",\n'
        '  "tone": "positivo|mixto|critico|no_disponible",\n'
        '  "key_points": ["punto 1", "punto 2", "punto 3"]\n'
        "}\n\n"
        f"Libro: {book_title}\n"
        f"Fuentes:\n{source_block}"
    )

    try:
        raw = llm.invoke(prompt).content
        data = _extract_json_object(raw if isinstance(raw, str) else str(raw))
        if isinstance(data, dict):
            return {
                "status": str(data.get("status", "fuentes_limitadas")),
                "summary": str(data.get("summary", "")).strip(),
                "tone": str(data.get("tone", "no_disponible")),
                "key_points": [str(x).strip() for x in data.get("key_points", []) if str(x).strip()],
            }
    except Exception:
        pass

    return {
        "status": "fuentes_limitadas",
        "summary": "No se pudo generar un resumen estructurado automáticamente con las fuentes disponibles.",
        "tone": "no_disponible",
        "key_points": [],
    }


def build_book_press_record(book: Dict[str, Any], llm: ChatOpenAI, fetch_external: bool, run_web_search: bool) -> Dict[str, Any]:
    title = book["title"]
    author = book["author"]
    year = book["year"]
    metadata = book["metadata"]

    web_catalog = metadata.get("web_catalog") if isinstance(metadata.get("web_catalog"), dict) else {}
    desc_short = str(web_catalog.get("web_short_description") or "")
    desc_long = str(web_catalog.get("web_description") or "")

    sources: List[PressSource] = []

    # Local editorial ficha sources
    if desc_short.strip():
        sources.append(
            PressSource(
                source_type="ficha_editorial",
                title="Descripción corta editorial",
                url="",
                domain="editorialdahbar.com",
                snippet=_html_to_text(desc_short, max_chars=700),
            )
        )

    if desc_long.strip():
        sources.append(
            PressSource(
                source_type="ficha_editorial",
                title="Descripción editorial",
                url="",
                domain="editorialdahbar.com",
                snippet=_html_to_text(desc_long, max_chars=1300),
            )
        )

    # External links found inside editorial description
    links = _extract_links_from_html(desc_long) if desc_long else []

    if run_web_search:
        for search_url in _search_web(title, author, max_results=3):
            if not any(item["url"] == search_url for item in links):
                links.append({"url": search_url, "title": "resultado_web"})

    # Collect external snippets
    seen = set()
    for item in links:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)

        domain = urlparse(url).netloc.lower()
        if _is_noisy_internal_catalog_link(url):
            continue

        if fetch_external:
            snippet = _fetch_web_text(url, timeout=12)
        else:
            snippet = ""

        if not snippet and "editorialdahbar.com" not in domain:
            continue

        sources.append(
            PressSource(
                source_type=_source_type_from_url(url),
                title=item.get("title") or "Fuente web",
                url=url,
                domain=domain,
                snippet=snippet[:1500] if snippet else "",
            )
        )

        # gentle pacing for external requests
        if fetch_external:
            time.sleep(0.2)

    # Deduplicate near-identical snippets
    unique_sources: List[PressSource] = []
    snippet_seen = set()
    for src in sources:
        key = (src.domain, src.url, src.snippet[:180])
        if key in snippet_seen:
            continue
        snippet_seen.add(key)
        unique_sources.append(src)

    summary = _summarize_press(title, unique_sources, llm)

    return {
        "book_title": title,
        "book_author": author,
        "book_year": year,
        "source_key": book["key"],
        "sources_count": len(unique_sources),
        "sources": [
            {
                "source_type": src.source_type,
                "title": src.title,
                "url": src.url,
                "domain": src.domain,
                "snippet": src.snippet,
            }
            for src in unique_sources
        ],
        "press_critique": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and summarize press/critique signals per book.")
    parser.add_argument("--max-books", type=int, default=0, help="Limit books processed (0 = all)")
    parser.add_argument("--no-fetch", action="store_true", help="Do not fetch external URLs")
    parser.add_argument("--web-search", action="store_true", help="Enable web search per book (DuckDuckGo HTML)")
    args = parser.parse_args()

    metadata = _load_metadata()
    books = _dedupe_books_by_title(metadata)
    books.sort(key=lambda b: b["title"].lower())

    if args.max_books > 0:
        books = books[: args.max_books]

    print(f"[PRESS] Books to process: {len(books)}")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    records: List[Dict[str, Any]] = []
    for idx, book in enumerate(books, start=1):
        print(f"[PRESS] {idx}/{len(books)} -> {book['title']}")
        record = build_book_press_record(
            book=book,
            llm=llm,
            fetch_external=not args.no_fetch,
            run_web_search=args.web_search,
        )
        records.append(record)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_books": len(records),
        "config": {
            "fetch_external": not args.no_fetch,
            "web_search": bool(args.web_search),
            "model": LLM_MODEL,
        },
        "books": records,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PRESS] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
