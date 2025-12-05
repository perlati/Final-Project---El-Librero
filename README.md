# 📚 El Librero · Editorial Dahbar AI Copilot

**Asistente inteligente de catálogo para Editorial Dahbar**

Sistema RAG (Retrieval-Augmented Generation) que permite consultar el catálogo de libros mediante preguntas en lenguaje natural. Diseñado para el equipo editorial de Editorial Dahbar, una editorial independiente venezolana especializada en ensayo, reportaje periodístico y crónicas.

---

## 🎯 Características principales

### Funcionalidades actuales (Fase 1)

- **Búsqueda temática en catálogo**: Encuentra todos los libros relevantes sobre un tema específico
- **Resúmenes de libros**: Genera resúmenes de 3 puntos clave de cualquier libro del catálogo
- **Deduplicación inteligente**: Agrupa resultados por libro para evitar duplicados
- **Book cards**: Presenta hasta 15 libros con snippets de contenido relevante
- **Interfaz moderna**: UI con marca Editorial Dahbar, alta legibilidad y configuración visible

### Capacidades técnicas

- **Retrieval avanzado**: MMR (Maximum Marginal Relevance) con k=40 chunks
- **Multi-query**: Genera variaciones de la pregunta para mejor recall
- **Metadata cleaning**: Limpia automáticamente nombres de autores y títulos
- **LangSmith tracking**: Monitoreo de uso de tokens y costos
- **Simple routing agent**: Sistema de selección de herramientas sin AgentExecutor

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI (app/)                       │
│              Interfaz web con marca Dahbar                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Editorial Agent (agents/)                      │
│         Keyword-based routing (NO AgentExecutor)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           │                        │
┌──────────▼─────────┐   ┌─────────▼──────────┐
│   search_books     │   │  summarize_book    │
│  (Catalogue-wide)  │   │  (Single book)     │
└──────────┬─────────┘   └─────────┬──────────┘
           │                        │
           └───────────┬────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 RAG Chains (rag/)                           │
│   • docs_to_book_cards(): Groups by normalized title       │
│   • Multi-query retrieval: 3 question variations           │
│   • Semantic search + fuzzy matching                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│            Chroma Vectorstore (vectorstore/)                │
│        2.6GB local DB • 73 books • ~1200-char chunks       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 OpenAI APIs                                 │
│   • gpt-4o-mini (generation)                               │
│   • text-embedding-3-small (embeddings)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Estructura del proyecto

```
editorialcopilot/
├── app/
│   ├── gradio_app.py           # UI principal con marca Dahbar
│   └── main_app.py              # [Legacy] Primera versión de UI
├── backend/
│   ├── config.py                # Variables de entorno y paths
│   ├── agents/
│   │   ├── editorial_agent.py   # Simple keyword-based router
│   │   └── tools.py             # 5 herramientas (2 activas, 3 stubs)
│   ├── rag/
│   │   ├── chains.py            # Book cards + summarize chains
│   │   └── retrievers.py        # [Legacy] Retrievers básicos
│   ├── ingestion/
│   │   ├── ingest_books.py      # Script de ingesta principal
│   │   ├── pdf_loader.py        # Carga y chunking de PDFs
│   │   └── extract_metadata_llm.py  # Extracción LLM de metadata
│   ├── vectorstore/
│   │   └── store.py             # Interface de Chroma
│   └── evaluation/
│       ├── run_eval.py          # Script de evaluación
│       └── eval_questions.json  # 10 preguntas de prueba
├── data/
│   ├── books/                   # 73 PDFs del catálogo
│   └── books_metadata_llm.json  # Metadata extraída por GPT
├── vectorstore/                 # Base de datos Chroma (2.6GB)
├── requirements.txt             # Dependencias Python
└── requirements.md              # Documento de especificaciones
```

---

## 🚀 Instalación y uso

### Requisitos previos

- Python 3.11+ (recomendado 3.12)
- Cuenta OpenAI con API key
- (Opcional) Cuenta LangSmith para tracking

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd editorialcopilot
```

### 2. Crear entorno virtual

```bash
# Con conda (recomendado)
conda create -n editorialcopilot python=3.12
conda activate editorialcopilot

# O con venv
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# LangSmith (opcional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=editorialcopilot
```

### 5. Lanzar la aplicación

```bash
python -m app.gradio_app
```

La interfaz estará disponible en: **http://127.0.0.1:7860**

---

## 🔧 Flujos de trabajo

### Re-ingestar el catálogo

Si añades nuevos PDFs a `data/books/` o actualizas metadatos:

```bash
# 1. (Opcional) Extraer metadata con GPT (tarda ~30min para 73 libros)
python -m backend.ingestion.extract_metadata_llm

# 2. Limpiar vectorstore anterior
rm -rf vectorstore/*

# 3. Ingestar libros
python -m backend.ingestion.ingest_books
```

### Ejecutar evaluación

```bash
python -m backend.evaluation.run_eval
```

Ejecuta las 10 preguntas de `backend/evaluation/eval_questions.json` y muestra las respuestas.

---

## 🛠️ Herramientas disponibles

### Activas (Fase 1)

| Herramienta | Descripción | Uso |
|------------|-------------|-----|
| **search_books** | Búsqueda temática en catálogo | "¿Qué libros tratan de política venezolana?" |
| **summarize_book** | Resumen de libro específico | "Resume las ideas centrales de 'Un dragón en el trópico'" |

### Planificadas (Fases 2-4)

| Herramienta | Fase | Descripción |
|------------|------|-------------|
| **search_media** | Fase 2 | Búsqueda en reseñas, entrevistas y prensa |
| **search_contracts** | Fase 3 | Consulta de cláusulas en contratos editoriales |
| **recommend_external_books** | Fase 4 | Recomendaciones de libros externos vía API |

---

## 💡 Ejemplos de uso

### Búsqueda temática

**Pregunta:**  
> ¿Qué libros del catálogo tratan de política venezolana?

**Respuesta:**  
Lista de 10-15 libros relevantes con explicaciones de por qué son relevantes.

---

### Resumen de libro

**Pregunta:**  
> Resume las ideas centrales del libro "Un dragón en el trópico"

**Respuesta:**  
- **Punto 1:** [Idea central extraída del libro]
- **Punto 2:** [Segunda idea principal]
- **Punto 3:** [Tercera idea clave]

---

### Comparación de libros

**Pregunta:**  
> Compara los libros del catálogo sobre economía latinoamericana

**Respuesta:**  
Análisis comparativo de múltiples libros sobre el tema.

---

## 📊 Especificaciones técnicas

### Modelo de lenguaje

- **LLM**: `gpt-4o-mini` (OpenAI)
- **Embeddings**: `text-embedding-3-small`
- **Vectorstore**: Chroma (local, 2.6GB)

### Retrieval

- **Algoritmo**: MMR (Maximum Marginal Relevance)
- **k**: 40 chunks por consulta
- **Chunk size**: 1200 caracteres
- **Chunk overlap**: 200 caracteres
- **Multi-query**: Habilitado (3 variaciones)
- **Deduplicación**: Por título normalizado

### Metadata

- **Fuente primaria**: LLM extraction (GPT-4)
- **Fuente secundaria**: Filename heuristics
- **Prioridad**: `llm_title > filename_title`
- **Limpieza**: Elimina ".indd", "Paper Back", "Tripa Final", etc.

### Costos estimados

Para un equipo de 5 usuarios con uso medio (40 consultas/día/usuario):

- **Consultas/mes**: 4,400 (22 días laborales)
- **Tokens/consulta**: ~1,500 (1,000 input + 500 output)
- **Costo mensual**: $5-10 USD

---

## 🎨 Interfaz de usuario

### Características de diseño

- **Marca**: Editorial Dahbar (negro #000000, dorado #d4af37)
- **Tipografía**: Georgia (serif, editorial)
- **Legibilidad**: Fuentes 16-18px, alto contraste
- **Layout**: 2 columnas (70/30) - consultas | configuración
- **Responsive**: Adaptable a diferentes tamaños de pantalla

### Secciones

1. **Header**: Logo y cita de Tomás Eloy Martínez
2. **Consulta**: Input de pregunta + 6 botones de ejemplo
3. **Respuesta**: Panel grande con respuesta del agente
4. **Historial**: Accordion colapsable con consultas previas
5. **Configuración**: Panel lateral con specs técnicas y estadísticas

---

## 📈 Mejoras planificadas

### Corto plazo

- [ ] Mejorar matching de títulos de libros (fuzzy search más robusto)
- [ ] Añadir filtros por año, autor, colección
- [ ] Implementar hybrid search (semántico + keyword)
- [ ] Re-ingestar vectorstore con metadata limpia
- [ ] Añadir números de página en citas

### Medio plazo (Fases 2-4)

- [ ] **Fase 2**: Ingestar reseñas, entrevistas y prensa
- [ ] **Fase 3**: Ingestar contratos y cláusulas
- [ ] **Fase 4**: Integrar API externa (Open Library)
- [ ] Añadir autenticación de usuarios
- [ ] Implementar query logging y analytics
- [ ] A/B testing de prompts

### Optimizaciones

- [ ] Comprimir prompts para reducir tokens
- [ ] Cachear consultas frecuentes
- [ ] Explorar LLMs locales (Ollama/llama.cpp)
- [ ] Backups automatizados del vectorstore

---

## 🐛 Problemas conocidos

### Metadata del vectorstore

El vectorstore actual contiene metadata con problemas:
- Algunos autores aparecen como "VENEZUELA EN EL" (extraídos del filename)
- Títulos con capitalización incorrecta ("RepúBlica BaldíA")
- Duplicados de PDFs con diferentes nombres

**Solución**: Re-ingestar con el script actualizado que prioriza metadata LLM.

### Deduplicación

Aunque la normalización de títulos reduce duplicados, algunos libros pueden aparecer múltiples veces si tienen títulos muy diferentes en diferentes PDFs.

**Solución implementada**: `_normalize_title_for_grouping()` en chains.py

---

## 📝 Licencia

Este proyecto es de uso interno para Editorial Dahbar.

**Editorial Dahbar**  
Editorial independiente venezolana  
Especializada en ensayo y reportaje periodístico  
🌐 [editorialdahbar.com](https://editorialdahbar.com)

---

## 👥 Equipo

Desarrollado para Editorial Dahbar como asistente interno de catálogo.

**Contacto:**  
📧 editorialdahbar@gmail.com  
📱 +58-212-7309873

---

## 🙏 Agradecimientos

- **LangChain**: Framework para RAG y agentes
- **OpenAI**: Modelos GPT-4 y embeddings
- **Chroma**: Vectorstore local de alto rendimiento
- **Gradio**: Framework de UI para ML
- **Editorial Dahbar**: Por la oportunidad de desarrollar esta herramienta

---

**Última actualización**: Diciembre 2025  
**Versión**: 1.0.0 (Fase 1 completa)
