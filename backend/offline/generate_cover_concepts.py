from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from backend.config import PROJECT_ROOT, LLM_MODEL

BRANDBOOK_JSON = Path(PROJECT_ROOT) / "data" / "website_media_curated" / "visual_brandbook.json"
OUTPUT_DIR = Path(PROJECT_ROOT) / "data" / "cover_generation"


def _load_brandbook() -> Dict[str, Any]:
    if not BRANDBOOK_JSON.exists():
        raise FileNotFoundError(
            f"Brandbook not found: {BRANDBOOK_JSON}. "
            "Run: python -m backend.offline.build_visual_brandbook"
        )
    return json.loads(BRANDBOOK_JSON.read_text(encoding="utf-8"))


def _cover_style_summary(brandbook: Dict[str, Any]) -> str:
    cover = brandbook.get("stats", {}).get("categories", {}).get("cover", {})
    layout = cover.get("layout_distribution", {})
    palette = [color for color, _ in cover.get("palette_top", [])[:8]]
    tokens = [token for token, _ in cover.get("top_name_tokens", [])[:15]]
    principles = brandbook.get("principles", [])

    chunks: List[str] = []
    if principles:
        chunks.append("Principios:\n- " + "\n- ".join(principles[:6]))
    if layout:
        chunks.append("Layout cover observado: " + ", ".join(f"{k}={v}" for k, v in layout.items()))
    if palette:
        chunks.append("Paleta dominante cover: " + ", ".join(palette))
    if tokens:
        chunks.append("Tokens visuales frecuentes: " + ", ".join(tokens))

    return "\n\n".join(chunks)


def _build_prompt_chain():
    system = (
        "Eres director/a de arte editorial para Editorial Dahbar.\n"
        "Generas conceptos de portada consistentes con el estilo visual observado en su histórico.\n"
        "No copies una tapa existente; crea una propuesta original pero coherente."
    )

    human = (
        "Contexto de estilo editorial:\n{style_context}\n\n"
        "Datos del libro:\n"
        "- Título: {title}\n"
        "- Subtítulo: {subtitle}\n"
        "- Autor: {author}\n"
        "- Género/tema: {theme}\n"
        "- Tono buscado: {tone}\n"
        "- Público objetivo: {audience}\n\n"
        "Produce exactamente estas secciones:\n"
        "1) Concepto rector (3-4 líneas)\n"
        "2) Dirección visual (paleta, tipografía, composición, textura, imagen central)\n"
        "3) Prompt final para generador de imagen (texto único, detallado, en español)\n"
        "4) Negative prompt (qué evitar para no romper el branding)\n"
        "5) Variantes A/B (2 mini-variantes con cambios claros)."
    )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.4)
    return prompt | llm | StrOutputParser()


def _slug(value: str) -> str:
    keep = []
    for ch in value.lower().strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "/", "\\"):
            keep.append("-")
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "cover-concept"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cover concepts aligned with Editorial Dahbar visual style.")
    parser.add_argument("--title", required=True, help="Book title")
    parser.add_argument("--subtitle", default="", help="Book subtitle")
    parser.add_argument("--author", required=True, help="Author name")
    parser.add_argument("--theme", required=True, help="Main theme / genre")
    parser.add_argument("--tone", default="sobrio, potente, contemporáneo", help="Desired tone")
    parser.add_argument("--audience", default="lectores de no ficción y debate público", help="Target audience")
    args = parser.parse_args()

    brandbook = _load_brandbook()
    style_context = _cover_style_summary(brandbook)

    chain = _build_prompt_chain()
    result = chain.invoke(
        {
            "style_context": style_context,
            "title": args.title,
            "subtitle": args.subtitle,
            "author": args.author,
            "theme": args.theme,
            "tone": args.tone,
            "audience": args.audience,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(args.title)
    output_path = OUTPUT_DIR / f"{slug}_cover_concept.md"

    header = [
        f"# Concepto de tapa – {args.title}",
        "",
        f"- Autor: {args.author}",
        f"- Tema: {args.theme}",
        f"- Tono: {args.tone}",
        f"- Público: {args.audience}",
        "",
    ]
    output_path.write_text("\n".join(header) + result + "\n", encoding="utf-8")

    print(f"[DONE] Cover concept saved to: {output_path}")


if __name__ == "__main__":
    main()
