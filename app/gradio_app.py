"""
El Librero - Editorial Dahbar AI Copilot

Modern Gradio UI for querying the publishing house's book catalogue.
Branded with Editorial Dahbar identity and advanced configuration options.
"""

import re
from typing import List, Tuple
import gradio as gr

from backend.agents.editorial_agent import agent_answer


def _extract_consulted_books(answer: str) -> str:
    """
    Parse a search_books answer and return a compact Markdown list of
    the books cited (title + year + author), for display in the
    'Libros consultados' sources panel.

    Handles lines like:
      - **Title (Year), Author** — explanation
      - Title (Year), Author — explanation
    """
    books: List[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        content = stripped[2:].strip()
        # Remove markdown bold markers
        content = re.sub(r"\*\*(.+?)\*\*", r"\1", content)
        # Take only the part before the em-dash explanation
        title_part = content.split(" — ")[0].split(" – ")[0].strip()
        if title_part:
            books.append(f"• {title_part}")
    if not books:
        return ""
    return "**📚 Libros consultados en esta respuesta:**\n\n" + "\n\n".join(books)


# Editorial Dahbar Brand Colors (from website)
BRAND_PRIMARY = "#1a1a1a"  # Dark charcoal (main text/headers)
BRAND_ACCENT = "#d4af37"   # Gold accent (inspired by literary excellence)
BRAND_BACKGROUND = "#f5f5f5"  # Light gray background
BRAND_SECONDARY = "#4a4a4a"  # Medium gray for secondary text

# Custom CSS for Editorial Dahbar branding
CUSTOM_CSS = """
/* Editorial Dahbar Branding - Maximum Legibility */
.gradio-container {
    font-family: 'Georgia', 'Times New Roman', serif !important;
    background: #ffffff !important;
    font-size: 16px !important;
}

/* Force all text to pure black */
* {
    color: #000000 !important;
}

/* Improved text legibility */
body, p, div, span, label, .markdown, .prose {
    color: #000000 !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* Header styling */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
    font-family: 'Georgia', serif !important;
    font-weight: 700 !important;
}

h1 { font-size: 2.8em !important; }
h2 { font-size: 2.2em !important; }
h3 { font-size: 1.8em !important; }

/* Primary button (search) */
.primary-btn {
    background: #000000 !important;
    border: 3px solid #d4af37 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    background: #d4af37 !important;
    color: #000000 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(212, 175, 55, 0.5) !important;
}

/* Example buttons */
.example-btn {
    background: #ffffff !important;
    border: 2px solid #000000 !important;
    color: #000000 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
}

.example-btn:hover {
    background: #d4af37 !important;
    color: #000000 !important;
    border-color: #d4af37 !important;
    transform: translateX(5px) !important;
}

/* Answer panel */
.answer-panel {
    background: #ffffff !important;
    border: 3px solid #000000 !important;
    border-left: 6px solid #d4af37 !important;
    padding: 25px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15) !important;
    border-radius: 8px !important;
    font-size: 17px !important;
    color: #000000 !important;
    line-height: 1.8 !important;
}

.answer-panel * {
    color: #000000 !important;
}

/* Config panel */
.config-panel {
    background: #ffffff !important;
    border: 2px solid #000000 !important;
    border-radius: 8px !important;
    padding: 18px !important;
    font-size: 17px !important;
    color: #000000 !important;
    line-height: 1.8 !important;
}

.config-panel * {
    color: #000000 !important;
    font-size: 17px !important;
    background: #ffffff !important;
}

.config-panel strong {
    font-weight: 700 !important;
    font-size: 18px !important;
}

/* Tool specs */
.tool-spec {
    background: #ffffff !important;
    border: 2px solid #000000 !important;
    border-left: 4px solid #d4af37 !important;
    padding: 15px !important;
    margin: 10px 0 !important;
    font-size: 17px !important;
    color: #000000 !important;
}

.tool-spec * {
    color: #000000 !important;
    font-size: 17px !important;
    background: #ffffff !important;
}

.tool-spec strong {
    color: #000000 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

.tool-spec em {
    color: #000000 !important;
    font-size: 16px !important;
    font-style: italic !important;
}

/* Literary quote styling */
.quote {
    font-style: italic !important;
    color: #000000 !important;
    background: #ffffff !important;
    border: 2px solid #000000 !important;
    border-left: 6px solid #d4af37 !important;
    padding: 20px !important;
    margin: 20px 0 !important;
    font-size: 17px !important;
    line-height: 1.7 !important;
}

.quote * {
    color: #000000 !important;
}

/* Input fields */
.input-box textarea, .input-box input, textarea, input {
    font-size: 16px !important;
    color: #000000 !important;
    background: #ffffff !important;
    border: 2px solid #000000 !important;
}

/* Accordion headers */
.accordion button {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    background: #000000 !important;
}

.accordion button span {
    color: #ffffff !important;
}

/* Accordion summary text */
summary, summary * {
    color: #ffffff !important;
    background: #000000 !important;
}

/* Markdown text in panels */
.markdown-text, .markdown, .prose {
    color: #000000 !important;
    font-size: 16px !important;
}

/* Links - keep gold for visibility but readable */
a {
    color: #000000 !important;
    font-weight: 600 !important;
    text-decoration: underline !important;
}

a:hover {
    color: #d4af37 !important;
}

/* Chatbot messages */
.message, .bot, .user {
    color: #000000 !important;
    background: #ffffff !important;
}

/* Labels */
label {
    color: #000000 !important;
    font-weight: 600 !important;
}
"""

# Example questions for quick access - designed for editorial team workflows
EXAMPLE_QUESTIONS = [
    # Catalogue overview
    "¿Qué libros del catálogo tratan de política venezolana?",
    # Acquisition/overlap check
    "¿Tenemos libros sobre transición democrática en América Latina?",
    # Book summary (specific title)
    "Resume las ideas centrales del libro 'Un dragón en el trópico'",
    # Author research
    "¿Qué autor del catálogo habla más sobre democracia y elecciones?",
    # Marketing/positioning
    "¿Qué libros recomendarías para promocionar a estudiantes de ciencias políticas?",
    # Theme comparison
    "Compara los libros del catálogo sobre economía latinoamericana",
]


def process_message(
    message: str,
    history: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], str, str, str]:
    """
    Process a user message and return updated chat state.

    Args:
        message: User's question.
        history: Current chat history as list of (user_msg, bot_msg) tuples.

    Returns:
        Tuple of (updated_history, answer_markdown, sources_markdown, empty_textbox)
    """
    if not message or not message.strip():
        return history, "", "", ""

    history = history or []
    history.append((message, "✓ Procesada"))

    try:
        answer = agent_answer(message, history[:-1])
        sources = _extract_consulted_books(answer)
        return history, answer, sources, ""

    except Exception as e:
        error_msg = (
            f"❌ **Error al consultar el catálogo**\n\n"
            f"`{type(e).__name__}: {str(e)}`"
        )
        return history, error_msg, "", ""


def clear_conversation() -> Tuple[List, str, str, str]:
    """
    Clear the conversation history, answer panel, and sources panel.

    Returns:
        Tuple of (empty_history, empty_answer, empty_sources, empty_textbox)
    """
    return [], "", "", ""


def build_app() -> gr.Blocks:
    """
    Build and return the Gradio Blocks app for El Librero.
    Modern design with Editorial Dahbar branding and configuration options.
    
    Returns:
        Configured Gradio Blocks app
    """
    with gr.Blocks(title="El Librero · Editorial Dahbar", css=CUSTOM_CSS) as app:
        
        # Header with Editorial Dahbar branding
        gr.Markdown(
            """
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%); color: white; border-bottom: 3px solid #d4af37; margin-bottom: 20px;'>
                <h1 style='color: white !important; margin: 0; font-size: 2.5em;'>📚 El Librero</h1>
                <p style='color: #d4af37; font-size: 1.2em; margin: 10px 0 0 0; font-weight: 600;'>Editorial Dahbar · Asistente de Catálogo Inteligente</p>
            </div>
            
            <div class='quote'>
                <em>"El talento narrativo es un atributo que puede dar a la información su más luminosa visibilidad."</em>
                <br><strong>— Tomás Eloy Martínez</strong>
            </div>
            """
        )
        
        with gr.Row():
            # Left column: Main interface
            with gr.Column(scale=7):
                # Question input
                gr.Markdown("### ❓ Consulta al catálogo")
                
                with gr.Row():
                    textbox = gr.Textbox(
                        label="",
                        placeholder="Ejemplo: '¿Qué libros tratan de política venezolana?' o 'Resume el libro Un dragón en el trópico'",
                        scale=5,
                        lines=2,
                        elem_classes="input-box"
                    )
                    send_btn = gr.Button("🔍 Buscar", variant="primary", scale=1, size="lg", elem_classes="primary-btn")
                
                # Example questions
                gr.Markdown("**💡 Consultas sugeridas:**")
                example_buttons = []
                with gr.Row():
                    for i in range(3):
                        btn = gr.Button(EXAMPLE_QUESTIONS[i], size="sm", elem_classes="example-btn")
                        example_buttons.append(btn)
                
                with gr.Row():
                    for i in range(3, 6):
                        btn = gr.Button(EXAMPLE_QUESTIONS[i], size="sm", elem_classes="example-btn")
                        example_buttons.append(btn)
                
                gr.Markdown("---")
                
                # Answer panel
                gr.Markdown("### 📖 Respuesta")
                answer_panel = gr.Markdown(
                    value="",
                    elem_classes="answer-panel"
                )

                # Sources panel — populated from parsed answer bullet points
                sources_panel = gr.Markdown(
                    value="",
                    elem_classes="config-panel"
                )

                gr.Markdown("---")
                
                # Compact chat history
                with gr.Accordion("📝 Historial de consultas", open=False):
                    chatbot = gr.Chatbot(
                        label="",
                        height=180,
                        show_label=False,
                    )
                    clear_btn = gr.Button("🗑️ Limpiar historial", size="sm", variant="secondary")
            
            # Right column: Tool specs and configuration
            with gr.Column(scale=3):
                gr.Markdown("### ⚙️ Configuración del sistema")
                
                with gr.Accordion("🛠️ Herramientas disponibles", open=True):
                    gr.Markdown(
                        """
                        <div class='tool-spec'>
                        <strong>1. search_books</strong> (Activa)<br>
                        📚 Búsqueda temática en catálogo<br>
                        <em>BM25 + MMR · multi-query · reranking · hasta 15 libros</em>
                        </div>
                        
                        <div class='tool-spec'>
                        <strong>2. summarize_book</strong> (Activa)<br>
                        📝 Resumen de libro específico<br>
                        <em>Output: 3 ideas centrales</em>
                        </div>
                        
                        <div class='tool-spec'>
                        <strong>3. search_media</strong> (Fase 2)<br>
                        📰 Reseñas y entrevistas<br>
                        <em>Estado: Planificado</em>
                        </div>
                        
                        <div class='tool-spec'>
                        <strong>4. search_contracts</strong> (Fase 3)<br>
                        📄 Consulta de contratos<br>
                        <em>Estado: Planificado</em>
                        </div>
                        
                        <div class='tool-spec'>
                        <strong>5. recommend_external_books</strong> (Fase 4)<br>
                        🌐 Recomendaciones externas<br>
                        <em>Estado: Planificado</em>
                        </div>
                        """,
                        elem_classes="config-panel"
                    )
                
                with gr.Accordion("🎛️ Parámetros avanzados", open=False):
                    gr.Markdown(
                        """
                        **Modelo LLM:**  
                        `gpt-4o-mini` (OpenAI)
                        
                        **Embeddings:**  
                        `text-embedding-3-small`
                        
                        **Vectorstore:**  
                        Chroma (local, 2.6GB)
                        
                        **Chunks totales:**  
                        ~73 libros procesados
                        
                        **Retrieval:**
                        - BM25 (30 %) + MMR (70 %) híbrido
                        - k=40 chunks por consulta
                        - Multi-query: Habilitado (2 variantes)
                        - Reranking LLM: Habilitado
                        - Deduplicación: Por título normalizado
                        
                        **Tracking:**  
                        LangSmith habilitado
                        """,
                        elem_classes="config-panel"
                    )
                
                with gr.Accordion("📊 Estadísticas", open=False):
                    gr.Markdown(
                        """
                        **Catálogo:**  
                        73 libros ingresados
                        
                        **Cobertura temática:**  
                        - Política venezolana
                        - Ciencias sociales
                        - Historia contemporánea
                        - Economía latinoamericana
                        - Testimonios y memorias
                        
                        **Última actualización:**  
                        Diciembre 2025
                        """,
                        elem_classes="config-panel"
                    )
                
                gr.Markdown("---")
                
                gr.Markdown(
                    """
                    <div style='text-align: center; padding: 15px; background: white; border-radius: 8px;'>
                    <strong style='color: #1a1a1a;'>Editorial Dahbar</strong><br>
                    <span style='color: #4a4a4a; font-size: 0.9em;'>Editorial independiente venezolana</span><br>
                    <span style='color: #4a4a4a; font-size: 0.9em;'>Especializada en ensayo y reportaje</span><br>
                    <a href='https://editorialdahbar.com' target='_blank' style='color: #d4af37; text-decoration: none;'>🌐 editorialdahbar.com</a>
                    </div>
                    """
                )
        
        # Wire up event handlers
        
        # Send message on button click
        send_btn.click(
            fn=process_message,
            inputs=[textbox, chatbot],
            outputs=[chatbot, answer_panel, sources_panel, textbox],
        )

        # Send message on Enter key
        textbox.submit(
            fn=process_message,
            inputs=[textbox, chatbot],
            outputs=[chatbot, answer_panel, sources_panel, textbox],
        )

        # Clear conversation
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chatbot, answer_panel, sources_panel, textbox],
        )

        # Example buttons - load example text and auto-send
        for i, btn in enumerate(example_buttons):
            example_text = EXAMPLE_QUESTIONS[i]
            btn.click(
                fn=lambda text=example_text: text,
                inputs=[],
                outputs=[textbox],
            ).then(
                fn=process_message,
                inputs=[textbox, chatbot],
                outputs=[chatbot, answer_panel, sources_panel, textbox],
            )
    
    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
