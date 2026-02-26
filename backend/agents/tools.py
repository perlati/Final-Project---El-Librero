# backend/agents/tools.py

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pathlib import Path
import json
import os
import re
from difflib import SequenceMatcher

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from backend.rag.chains import build_books_rag_chain, build_book_summary_chain, build_deep_insight_chain
from backend.config import BOOKS_COLLECTION, LLM_MODEL, PROJECT_ROOT
from backend.vectorstore.store import get_vectorstore
from backend.utils.text import normalize_title

# Build chains once at startup.
# Hybrid retrieval (BM25 + MMR) + multi-query + LLM reranking give the
# best catalogue recall.  The vectorstore must be populated before import.
_books_rag_chain = build_books_rag_chain(
    use_multi_query=True,
    use_hybrid=True,
    use_reranking=True,
)
summary_chain = build_book_summary_chain()
deep_insight_chain = build_deep_insight_chain()
BOOK_REPORTS_DIR = Path(PROJECT_ROOT) / "data" / "book_reports"
BOOK_PRESS_DB_PATH = Path(PROJECT_ROOT) / "data" / "books_press_critique.json"


def _infer_book_title_from_question(question: str) -> str | None:
    """
    Infer a target catalogue book when the user asks for a summary
    without explicitly naming a title.

    Strategy:
    - Retrieve relevant catalog chunks for the question.
    - Aggregate by book_title with frequency + most recent year.
    - Prefer recent books while keeping semantic relevance.
    """
    if not question or not question.strip():
        return None

    vs = get_vectorstore(BOOKS_COLLECTION)
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 30,
            "filter": {"doc_type": "catalog_book"},
        },
    )

    docs = retriever.invoke(question)
    if not docs:
        return None

    candidates = {}
    for doc in docs:
        metadata = doc.metadata or {}
        title = (metadata.get("book_title") or "").strip()
        if not title:
            continue

        raw_year = metadata.get("pub_year")
        try:
            year = int(raw_year) if raw_year is not None else 0
        except (ValueError, TypeError):
            year = 0

        entry = candidates.setdefault(title, {"freq": 0, "year": 0})
        entry["freq"] += 1
        if year > entry["year"]:
            entry["year"] = year

    if not candidates:
        return None

    # Sort by recency first, then by retrieval frequency.
    best_title = max(candidates.items(), key=lambda item: (item[1]["year"], item[1]["freq"]))[0]
    return best_title


def _load_book_report_text(book_title: str) -> str | None:
    normalized = normalize_title(book_title).lower()
    if not normalized:
        return None
    path = BOOK_REPORTS_DIR / f"{normalized}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _answer_with_book_report(report: str, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Eres El Librero.\n"
                    "Tienes un informe de lectura completo de un libro del catálogo.\n"
                    "Responde con profundidad usando solo la evidencia del informe.\n"
                    "Si algo no está soportado por el informe, dilo explícitamente.\n"
                    "Responde siempre en español."
                ),
            ),
            (
                "human",
                "Informe de lectura:\n{report}\n\n"
                "Pregunta del usuario:\n{question}\n\n"
                "Responde con máxima claridad y profundidad editorial."
            ),
        ]
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"report": report, "question": question})


def _load_press_db() -> dict:
    if not BOOK_PRESS_DB_PATH.exists():
        return {}
    try:
        data = json.loads(BOOK_PRESS_DB_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_explicit_title(query: str) -> str | None:
    patterns = [
        r"['\"]([^'\"]{3,120})['\"]",
        r"sobre\s+([\w\sáéíóúüñÁÉÍÓÚÜÑ\-¿?¡!]{4,120})",
        r"de\s+([\w\sáéíóúüñÁÉÍÓÚÜÑ\-¿?¡!]{4,120})",
    ]
    for pattern in patterns:
        match = re.search(pattern, query or "", flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" .,:;!?¿¡")
            if len(value) >= 4:
                return value
    return None


def _best_press_record(query: str, records: list[dict]) -> dict | None:
    if not records:
        return None

    explicit = _extract_explicit_title(query or "")
    normalized_query = normalize_title(explicit or query or "").lower()
    if not normalized_query:
        return None

    best = None
    best_score = 0.0
    for record in records:
        title = normalize_title(record.get("book_title", "")).lower()
        if not title:
            continue

        if title in normalized_query or normalized_query in title:
            score = 1.0
        else:
            score = SequenceMatcher(None, normalized_query, title).ratio()

        if score > best_score:
            best_score = score
            best = record

    if best_score < 0.45:
        return None
    return best


def _answer_press_query(record: dict, query: str) -> str:
    sources = record.get("sources", [])
    source_block = "\n\n".join(
        [
            f"- Tipo: {src.get('source_type', '')} | Dominio: {src.get('domain', '')} | URL: {src.get('url', '')}\n"
            f"  Extracto: {(src.get('snippet', '') or '')[:900]}"
            for src in sources[:8]
        ]
    )

    base_summary = record.get("press_critique", {}).get("summary", "")
    tone = record.get("press_critique", {}).get("tone", "no_disponible")
    key_points = record.get("press_critique", {}).get("key_points", [])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Eres analista editorial de prensa y recepción crítica. "
                    "Responde SOLO con la evidencia dada. "
                    "Si faltan fuentes, dilo explícitamente. "
                    "Siempre en español."
                ),
            ),
            (
                "human",
                "Libro: {title}\n"
                "Resumen base: {base_summary}\n"
                "Tono estimado: {tone}\n"
                "Puntos clave: {key_points}\n"
                "Fuentes disponibles:\n{sources}\n\n"
                "Pregunta: {query}\n\n"
                "Entrega:\n"
                "1) Respuesta directa (4-8 líneas).\n"
                "2) Evidencia (2-5 bullets con fuente: dominio o URL).\n"
                "3) Nivel de confianza: alto/medio/bajo y por qué."
            ),
        ]
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "title": record.get("book_title", ""),
            "base_summary": base_summary,
            "tone": tone,
            "key_points": "; ".join(key_points),
            "sources": source_block,
            "query": query,
        }
    )


@tool
def search_books(query: str) -> str:
    """
    Search the internal catalogue for books relevant to a topic or question.
    
    Use this when the user asks about THEMES, TOPICS or TYPES OF BOOKS.
    Returns a comprehensive list of all relevant catalogue books.
    """
    if not query or not query.strip():
        return "No se proporcionó una pregunta válida sobre el catálogo."

    return _books_rag_chain.invoke(query)


@tool
def summarize_book_tool(book_title: str, question: str = "Resume este libro en 10 viñetas") -> str:
    """
    Summarize ONE specific catalogue book in exactly 10 bullet points.

    Use this when the user explicitly asks for the main ideas, summary,
    or key points of a single book, e.g.:
    - "Dame ideas centrales de 'Un dragón en el trópico'."
    - "Resume en diez puntos el libro 'Libres'."

    Args:
        book_title: The target book title (approximate title is allowed).
        question: Optional user question context.

    Returns exactly 10 bullet points in Spanish.
    """
    resolved_title = (book_title or "").strip()
    if not resolved_title:
        inferred_title = _infer_book_title_from_question(question)
        if inferred_title:
            resolved_title = inferred_title

    if not resolved_title:
        return "No pude identificar un libro del catálogo para resumir con esta pregunta."

    return summary_chain.invoke({
        "book_title": resolved_title,
        "question": question.strip() if question else "Resume este libro en 10 viñetas",
    })


@tool
def deep_insights_book_tool(book_title: str, question: str = "Dame un informe de lectura completo de este libro") -> str:
    """
    Generate a deep editorial reading report for ONE catalogue book.

    Priority order:
    1) Use offline report from data/book_reports if present.
    2) Fallback to deep RAG chain over book chunks + metadata.
    """
    resolved_title = (book_title or "").strip()
    if not resolved_title:
        inferred_title = _infer_book_title_from_question(question)
        if inferred_title:
            resolved_title = inferred_title

    if not resolved_title:
        return "No pude identificar un libro del catálogo para generar el informe de lectura."

    report = _load_book_report_text(resolved_title)
    if report:
        return _answer_with_book_report(report, question.strip() if question else "")

    return deep_insight_chain.invoke({
        "book_title": resolved_title,
        "question": question.strip() if question else "Dame un informe de lectura completo de este libro",
    })


@tool
def search_media(query: str) -> str:
    """
    Busca señales de prensa, entrevistas y crítica por libro desde la base
    `data/books_press_critique.json` generada offline.
    """
    db = _load_press_db()
    records = db.get("books", []) if isinstance(db, dict) else []
    if not records:
        return (
            "No hay base de prensa/crítica cargada todavía.\n"
            "Ejecuta: python -m backend.ingestion.collect_book_press_critique --web-search"
        )

    q = (query or "").strip()
    lower_q = q.lower()
    asks_overview = any(token in lower_q for token in ["catálogo", "catalogo", "todos", "general", "panorama"])

    if asks_overview:
        with_sources = [r for r in records if int(r.get("sources_count", 0)) > 0]
        top = sorted(with_sources, key=lambda x: int(x.get("sources_count", 0)), reverse=True)[:12]
        lines = [
            "Panorama de prensa/crítica del catálogo:",
            f"- Libros analizados: {len(records)}",
            f"- Libros con alguna fuente: {len(with_sources)}",
            "- Top libros con mayor cobertura:",
        ]
        for item in top:
            title = item.get("book_title", "")
            count = item.get("sources_count", 0)
            tone = item.get("press_critique", {}).get("tone", "no_disponible")
            lines.append(f"  - {title} ({count} fuentes, tono: {tone})")
        return "\n".join(lines)

    best = _best_press_record(q, records)
    if not best:
        return (
            "No pude mapear tu pregunta a un libro concreto en la base de prensa/crítica.\n"
            "Prueba incluyendo el título exacto entre comillas."
        )

    return _answer_press_query(best, q)


@tool
def search_contracts(query: str) -> str:
    """
    [Fase 3 – contratos]

    En una versión futura, esta herramienta buscaría cláusulas relevantes
    en contratos de edición, distribución y cesión de derechos.

    Ahora mismo actúa como stub y siempre recuerda que no es asesoría legal.
    """
    return (
        "Fase 3 – contratos (prototipo):\n"
        "Esta herramienta analizaría contratos de edición, distribución y cesión "
        "de derechos para encontrar cláusulas relevantes (territorio, duración, "
        "royalties, reversión, etc.).\n\n"
        "Aviso: incluso en la versión completa, las respuestas serían sólo apoyo "
        "interno para el equipo editorial y no sustituyen asesoría legal profesional."
    )


@tool
def recommend_external_books(query: str) -> str:
    """
    [Fase 4 – recomendaciones externas]

    En una versión futura, esta herramienta llamaría a una API de libros abierta
    (por ejemplo, Open Library) para recomendar títulos externos alineados con la
    línea editorial.

    De momento devuelve una explicación estática.
    """
    return (
        "Fase 4 – recomendaciones externas (prototipo):\n"
        "Aquí conectaríamos con una API de libros (por ejemplo Open Library) para "
        "buscar títulos externos que encajen con tu consulta y con la línea editorial.\n\n"
        "La versión completa podría:\n"
        "- Filtrar por idioma y región.\n"
        "- Puntuar afinidad temática con el catálogo propio.\n"
        "- Proponer títulos para adquisición o coedición."
    )
