from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import time
from typing import Dict, List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from backend.vectorstore.store import get_all_documents
from backend.config import PROJECT_ROOT, BOOKS_COLLECTION, LLM_MODEL
from backend.utils.text import normalize_title

BOOK_REPORTS_DIR = Path(PROJECT_ROOT) / "data" / "book_reports"
PROGRESS_DIR = BOOK_REPORTS_DIR / "_progress"
METADATA_PATH = Path(PROJECT_ROOT) / "data" / "books_metadata_llm.json"


def load_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_PATH.exists():
        return {}

    data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            nt = normalize_title(item.get("normalized_title") or item.get("title", "")).lower()
            if nt:
                out[nt] = item
    elif isinstance(data, dict):
        for _, item in data.items():
            if not isinstance(item, dict):
                continue
            nt = normalize_title(item.get("normalized_title") or item.get("title", "")).lower()
            if nt:
                out[nt] = item
    return out


def _meta_field(meta: Any, *keys: str) -> str:
    if not isinstance(meta, dict):
        return ""
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            txt = str(value).strip()
            if txt:
                return txt
    return ""


def group_docs_by_book() -> Dict[str, List[Any]]:
    """Group all catalog documents by normalized title."""
    docs = get_all_documents(BOOKS_COLLECTION, doc_type="catalog_book")
    grouped: Dict[str, List[Any]] = defaultdict(list)

    for doc in docs:
        meta = doc.metadata or {}
        nt = normalize_title(meta.get("normalized_title") or meta.get("book_title") or "").lower()
        if not nt:
            continue
        grouped[nt].append(doc)

    for nt, lst in grouped.items():
        lst.sort(key=lambda d: (d.metadata or {}).get("page") or 0)

    return grouped


def chunk_book_text(docs: List[Any], max_chars: int = 6000) -> List[str]:
    """Concatenate book text and split into segments of ~max_chars."""
    segments: List[str] = []
    current = ""

    for doc in docs:
        page = (doc.metadata or {}).get("page", "?")
        section = (doc.metadata or {}).get("section_type", "body")
        text = f"[Página: {page} | Sección: {section}]\n{doc.page_content}".strip()

        if len(current) + len(text) <= max_chars:
            current = f"{current}\n\n{text}" if current else text
        else:
            if current:
                segments.append(current)
            current = text

    if current:
        segments.append(current)

    return segments


def build_segment_summarizer():
    system = (
        "Eres un lector profesional.\n"
        "Tu tarea es resumir un segmento de un libro para luego construir un informe de lectura completo.\n\n"
        "Reglas:\n"
        "- No intentes describir todo el libro, solo este segmento.\n"
        "- Extrae ideas, argumentos, temas, tono y pistas de audiencia.\n"
        "- Sé concreto, en 5–8 viñetas máximo."
    )
    human = (
        "Segmento del libro:\n\n"
        "{segment}\n\n"
        "Escribe un resumen en viñetas de alto nivel de este segmento,\n"
        "pensando en que luego se combinará con otros segmentos."
    )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()


def build_reduce_summarizer():
    system = (
        "Eres El Librero, lector profesional y editor senior.\n\n"
        "Has recibido varios resúmenes parciales de un libro y su ficha editorial.\n"
        "Tu tarea es producir un informe de lectura completo y profundo, usable por un equipo editorial."
    )
    human = (
        "Título: {title}\n"
        "Autor(es): {author}\n"
        "Año: {year}\n"
        "Ficha web / descripción:\n"
        "{ficha}\n\n"
        "Resúmenes parciales de segmentos del libro:\n"
        "{segment_summaries}\n\n"
        "Ahora escribe un informe de lectura completo con esta estructura:\n\n"
        "1. Tesis central del libro\n"
        "2. 4–6 ideas clave\n"
        "3. Estructura y recorrido (cómo avanza el libro)\n"
        "4. Temas y marcos conceptuales\n"
        "5. Valor editorial y posibles ángulos de venta\n"
        "6. Tipo de lector ideal y posibles advertencias (densidad, tecnicismo, tono)\n\n"
        "Sé preciso, profundo y concreto.\n"
        "Si algo lo infieres (no está explícito), márcalo como interpretación."
    )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()


def build_batch_reducer():
    system = (
        "Eres un editor que consolida resúmenes parciales de un mismo libro.\n"
        "Debes comprimir información sin perder ideas clave."
    )
    human = (
        "Resúmenes parciales:\n"
        "{partial_summaries}\n\n"
        "Devuelve una síntesis compacta en 12–18 viñetas, preservando:\n"
        "- argumentos centrales\n"
        "- estructura/recorrido\n"
        "- temas y marcos conceptuales\n"
        "- tono y tipo de lector\n"
        "No inventes datos."
    )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()


def _safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("\n", " ")
        .strip()
    )


def _progress_path(normalized_title: str) -> Path:
    return PROGRESS_DIR / f"{_safe_name(normalized_title)}.json"


def _load_progress(normalized_title: str, total_segments: int) -> List[str]:
    path = _progress_path(normalized_title)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    summaries = payload.get("segment_summaries", []) if isinstance(payload, dict) else []
    if not isinstance(summaries, list):
        return []
    cleaned = [str(item) for item in summaries if str(item).strip()]
    return cleaned[:total_segments]


def _save_progress(normalized_title: str, segment_summaries: List[str]) -> None:
    path = _progress_path(normalized_title)
    payload = {"segment_summaries": segment_summaries}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _invoke_with_retries(chain, payload: Dict[str, Any], retries: int = 3, sleep_s: float = 2.0) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            wait = sleep_s * attempt
            print(f"    [WARN] Retry {attempt}/{retries} after error: {type(exc).__name__}")
            time.sleep(wait)
    if last_error:
        raise last_error
    raise RuntimeError("Unknown invocation failure")


def _hierarchical_reduce(segment_summaries: List[str], batch_reducer, batch_size: int = 20) -> str:
    if not segment_summaries:
        return ""

    level = segment_summaries
    round_idx = 1

    while len(level) > 1:
        total_batches = math.ceil(len(level) / batch_size)
        print(f"  - Hierarchical reduce round {round_idx}: {len(level)} summaries in {total_batches} batches")
        next_level: List[str] = []

        for i in range(0, len(level), batch_size):
            batch = level[i:i + batch_size]
            merged = _invoke_with_retries(
                batch_reducer,
                {"partial_summaries": "\n\n".join(batch)},
                retries=3,
                sleep_s=2.0,
            )
            batch_number = (i // batch_size) + 1
            next_level.append(f"### Síntesis lote {batch_number}\n{merged}")

        level = next_level
        round_idx += 1

    return level[0]


def main() -> None:
    BOOK_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_by_title = load_metadata()
    grouped_docs = group_docs_by_book()
    seg_summarizer = build_segment_summarizer()
    batch_reducer = build_batch_reducer()
    reduce_summarizer = build_reduce_summarizer()
    failed_books: List[str] = []

    for nt, docs in grouped_docs.items():
        out_path = BOOK_REPORTS_DIR / f"{nt}.md"
        if out_path.exists():
            print(f"[SKIP] Report already exists for {nt}")
            continue

        meta = metadata_by_title.get(nt, {})
        web_catalog = meta.get("web_catalog") if isinstance(meta.get("web_catalog"), dict) else {}

        title = _meta_field(meta, "title") or nt
        author = _meta_field(meta, "author", "main_author", "book_author") or _meta_field(web_catalog, "web_author") or "Autor no identificado"
        year = _meta_field(meta, "year", "pub_year") or _meta_field(web_catalog, "web_edition")
        ficha = _meta_field(meta, "web_ficha", "description") or _meta_field(web_catalog, "web_description", "web_short_description")

        print(f"[REPORT] Generating report for {nt}...")

        try:
            segments = chunk_book_text(docs, max_chars=6000)
            segment_summaries = _load_progress(nt, len(segments))

            if segment_summaries:
                print(f"  - Resuming from segment {len(segment_summaries) + 1}/{len(segments)}")

            for i in range(len(segment_summaries), len(segments)):
                print(f"  - Summarizing segment {i + 1}/{len(segments)}")
                summary = _invoke_with_retries(
                    seg_summarizer,
                    {"segment": segments[i]},
                    retries=3,
                    sleep_s=1.5,
                )
                segment_summaries.append(f"### Segmento {i + 1}\n{summary}")
                _save_progress(nt, segment_summaries)

            if len(segment_summaries) > 40:
                print(f"  - Compressing {len(segment_summaries)} segment summaries before final report")
                summaries_for_final = _hierarchical_reduce(segment_summaries, batch_reducer, batch_size=20)
            else:
                summaries_for_final = "\n\n".join(segment_summaries)

            reduce_input = {
                "title": title,
                "author": author,
                "year": year,
                "ficha": ficha,
                "segment_summaries": summaries_for_final,
            }
            report = _invoke_with_retries(reduce_summarizer, reduce_input, retries=3, sleep_s=2.0)
            out_path.write_text(report, encoding="utf-8")

            progress_path = _progress_path(nt)
            if progress_path.exists():
                progress_path.unlink()

            print(f"[DONE] Saved report to {out_path}")

        except Exception as exc:  # noqa: BLE001
            failed_books.append(nt)
            print(f"[FAIL] {nt}: {type(exc).__name__}: {exc}")
            continue

    if failed_books:
        print("\n[SUMMARY] Failed books:")
        for book in failed_books:
            print(f"- {book}")


if __name__ == "__main__":
    main()
