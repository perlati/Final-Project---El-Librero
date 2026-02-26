# backend/agents/editorial_agent.py

from typing import List, Any
import re

from backend.agents.tools import (
    search_books,
    summarize_book_tool,
    deep_insights_book_tool,
    search_media,
    search_contracts,
    recommend_external_books,
)


def _history_to_text(history: List[Any]) -> str:
    """Convert Gradio history to text format."""
    lines = []
    
    for turn in history:
        if isinstance(turn, (tuple, list)) and len(turn) == 2:
            user_msg, bot_msg = turn
            if user_msg:
                lines.append(f"Usuario: {user_msg}")
            if bot_msg:
                lines.append(f"Asistente: {bot_msg}")
        elif isinstance(turn, dict):
            role = turn.get("role") or turn.get("speaker")
            content = turn.get("content") or turn.get("value")
            if content:
                if role in ("user", "human"):
                    lines.append(f"Usuario: {content}")
                elif role in ("assistant", "bot"):
                    lines.append(f"Asistente: {content}")
    
    return "\n".join(lines)


def _extract_book_title(text: str) -> str | None:
    """
    Try to extract a book title from a summary request.
    Handles patterns with/without quotes:
    - "del libro 'Title'", "resume 'Title'"
    - "resume Libres", "el libro Bifurcación"
    - "ideas centrales de Title"
    """
    def _is_generic_candidate(candidate: str) -> bool:
        normalized = candidate.strip().lower()
        if not normalized:
            return True

        generic_prefixes = [
            "uno de los libros",
            "un libro",
            "una obra",
            "algún libro",
            "algun libro",
            "los libros",
            "libros del catálogo",
            "libros del catalogo",
        ]
        if any(normalized.startswith(prefix) for prefix in generic_prefixes):
            return True

        if normalized.startswith("sobre "):
            return True

        if "del catálogo" in normalized or "del catalogo" in normalized:
            return True

        if "más recientes" in normalized or "mas recientes" in normalized:
            return True

        return False

    quoted_patterns = [
        r"['\"]([^'\"]+)['\"]",
    ]
    
    for pattern in quoted_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if not _is_generic_candidate(candidate):
                return candidate
    
    de_matches = re.findall(r"\bde\s+([^,.;!?]+)", text, re.IGNORECASE)
    if de_matches:
        title = de_matches[-1].strip(" .,!?:;")
        title = re.sub(r"^(?:lectura|analisis|análisis|informe)\s+de\s+", "", title, flags=re.IGNORECASE)
        if title and len(title) > 2 and not _is_generic_candidate(title):
            return title

    # Handles: "Resume en diez puntos Libres"
    resume_match = re.search(
        r"\b(?:resume|resumen(?:\s+de)?|resumir)\b(?:\s+en\s+\w+\s+\w+)?\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if resume_match:
        title = resume_match.group(1).strip(" .,!?:;")
        title = re.sub(r"^(?:el|la|los|las|libro)\s+", "", title, flags=re.IGNORECASE)
        if title and len(title) > 2 and not _is_generic_candidate(title):
            return title
    
    return None


def _select_tool_and_input(text: str) -> tuple:
    """
    Select the appropriate tool and prepare the input based on the user's question.
    
    Returns:
        Tuple of (tool_function, input_string)
    """
    lower = text.lower()

    deep_insight_keywords = [
        "informe de lectura",
        "análisis profundo",
        "analisis profundo",
        "análisis detallado",
        "analisis detallado",
        "lectura profunda",
    ]
    if any(keyword in lower for keyword in deep_insight_keywords):
        book_title = _extract_book_title(text)
        return (deep_insights_book_tool, {"book_title": book_title or "", "question": text})

    summary_keywords = ["ideas centrales", "resume", "resumen de"]
    
    if any(keyword in lower for keyword in summary_keywords):
        book_title = _extract_book_title(text)
        return (summarize_book_tool, {"book_title": book_title or "", "question": text})

    # Contratos / derechos
    if any(word in lower for word in ["contrato", "contratos", "derechos", "licencia", "licencias"]):
        return (search_contracts, text)

    # Media / reseñas / entrevistas
    if any(word in lower for word in ["entrevista", "reseña", "reseñas", "prensa", "medios", "media"]):
        return (search_media, text)

    # Recomendaciones externas
    if any(word in lower for word in ["recomienda", "recomiéndame", "recomendaciones", "externo", "externos"]):
        return (recommend_external_books, text)

    # Por defecto: catálogo de libros
    return (search_books, text)


def agent_answer(question: str, history: List[Any]) -> str:
    """
    Main entry point for the editorial agent.
    
    Selects the appropriate tool based on the question and invokes it.
    
    Args:
        question: User's question
        history: Conversation history
        
    Returns:
        Agent's response as a string
    """
    if not question or not question.strip():
        return "Por favor escribe una pregunta."
    
    try:
        # Select tool and prepare input
        tool, tool_input = _select_tool_and_input(question)
        
        # Invoke the tool
        response = tool.invoke(tool_input)
        
        return response
        
    except Exception as e:
        return f"Ocurrió un error al consultar el catálogo: `{type(e).__name__}: {str(e)}`"

