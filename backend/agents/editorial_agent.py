# backend/agents/editorial_agent.py

from typing import List, Any
import re

from backend.agents.tools import (
    search_books,
    summarize_book,
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
    # First try patterns with quotes (most specific)
    quoted_patterns = [
        r"del libro ['\"]([^'\"]+)['\"]",
        r"de ['\"]([^'\"]+)['\"]",
        r"libro ['\"]([^'\"]+)['\"]",
        r"resume ['\"]([^'\"]+)['\"]",
        r"resumen ['\"]([^'\"]+)['\"]",
        r"['\"]([^'\"]+)['\"]\\s*$",  # Title at end in quotes
    ]
    
    for pattern in quoted_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Then try patterns without quotes (less specific)
    # Match capitalized words after key phrases
    unquoted_patterns = [
        r"del libro\\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ\\s]+?)(?:\\s+(?:es|son|trata|habla|presenta)|[.,?!]|$)",
        r"resume(?:\\s+el\\s+libro)?\\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ\\s]+?)(?:\\s+(?:es|son|trata)|[.,?!]|$)",
        r"(?:ideas|puntos)\\s+(?:centrales|principales)\\s+de\\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ\\s]+?)(?:\\s+(?:es|son)|[.,?!]|$)",
        r"libro\\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ\\s]+?)(?:\\s+(?:es|son|trata)|[.,?!]|$)",
    ]
    
    for pattern in unquoted_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Filter out common false positives
            if title.lower() not in ['el', 'la', 'los', 'las', 'un', 'una', 'del', 'de']:
                return title
    
    return None


def _select_tool_and_input(text: str) -> tuple:
    """
    Select the appropriate tool and prepare the input based on the user's question.
    
    Returns:
        Tuple of (tool_function, input_string)
    """
    lower = text.lower()

    # Check for book summary requests (high priority)
    summary_keywords = [
        "ideas centrales",
        "ideas principales",
        "resume",
        "resumen",
        "resumir",
        "sintetiza",
        "síntesis",
        "puntos clave",
        "puntos principales",
        "tres ideas",
        "3 ideas",
        "principales temas",
        "mensaje principal",
    ]
    
    if any(keyword in lower for keyword in summary_keywords):
        # Try to extract book title
        book_title = _extract_book_title(text)
        if book_title:
            return (summarize_book, book_title)
        # If no title found, let search_books handle it
        return (search_books, text)

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

