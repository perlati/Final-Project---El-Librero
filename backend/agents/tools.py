# backend/agents/tools.py

from langchain_core.tools import tool

from backend.rag.chains import build_books_rag_chain, build_summarize_book_chain

# Build chains once at startup.
# Hybrid retrieval (BM25 + MMR) + multi-query + LLM reranking give the
# best catalogue recall.  The vectorstore must be populated before import.
_books_rag_chain = build_books_rag_chain(
    use_multi_query=True,
    use_hybrid=True,
    use_reranking=True,
)
_summarize_chain = build_summarize_book_chain()


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
def summarize_book(book_title: str) -> str:
    """
    Summarize the central ideas of ONE specific catalogue book.

    Use this when the user explicitly asks for the main ideas, summary,
    or key points of a single book, e.g.:
    - "Dame tres ideas centrales de 'Un dragón en el trópico'."
    - "Resume en tres puntos el libro 'Libres'."

    The argument must be the book title as it appears in the catalogue.
    Returns exactly 3 bullet points with the book's central ideas.
    """
    if not book_title or not book_title.strip():
        return "No se proporcionó un título de libro válido."
    
    return _summarize_chain.invoke(book_title)


@tool
def search_media(query: str) -> str:
    """
    [Fase 2 – media]

    En una versión futura, esta herramienta buscaría en notas de prensa,
    entrevistas y reseñas asociadas a los libros.

    De momento devuelve una explicación estática para mostrar la arquitectura.
    """
    return (
        "Fase 2 – media (prototipo):\n"
        "Aquí buscaríamos en entrevistas, reseñas, notas de prensa y otros "
        "materiales mediáticos vinculados a los libros.\n\n"
        "Por ahora esta herramienta es un stub pensado para demostrar que el "
        "sistema soporta múltiples tools. La implementación real incluiría:\n"
        "- Ingesta de PDFs/HTML de reseñas y entrevistas.\n"
        "- Un vectorstore separado para media.\n"
        "- Un RAG similar al de libros."
    )


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
