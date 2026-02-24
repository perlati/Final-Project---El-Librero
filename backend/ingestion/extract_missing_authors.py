#!/usr/bin/env python3
"""
Script para extraer metadata con LLM solo de libros sin autor
"""

import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import openai
from dotenv import load_dotenv
import os

load_dotenv()

#Configuración
PROJECT_ROOT = Path(__file__).parent.parent.parent
BOOKS_DIR = PROJECT_ROOT / "data" / "books"
METADATA_FILE = PROJECT_ROOT / "data" / "books_metadata_llm.json"

openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"


def extract_metadata_for_missing_authors():
    """Extrae metadata solo de libros sin autor"""
    
    # Cargar metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Filtrar solo archivos originales sin autor
    without_author = [
        filename for filename, meta in metadata.items()
        if not filename.startswith('97') and (not meta.get('main_author') or meta.get('main_author') == 'Desconocido')
    ]
    
    print(f"📚 Procesando {len(without_author)} libros sin autor...\n")
    
    for i, filename in enumerate(without_author, 1):
        print(f"[{i}/{len(without_author)}] {filename}")
        
        pdf_path = BOOKS_DIR / filename
        if not pdf_path.exists():
            print(f"  ⚠️  PDF no encontrado")
            continue
        
        try:
            # Leer primeras 10 páginas
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            num_pages = min(10, len(pages))
            text_chunks = []
            for j in range(num_pages):
                text = pages[j].page_content
                text_chunks.append(f"--- Página {j + 1} ---\n{text[:1500]}")
            
            full_text = "\n\n".join(text_chunks)
            
            # Llamada a LLM
            prompt = f"""Analiza las primeras páginas de este libro y extrae la metadata en formato JSON:

{{
  "title": "título completo sin 'Paper Back', 'Final', etc.",
  "main_author": "autor principal (NO editor, coordinador, fotógrafo)",
  "year": "año de publicación (4 dígitos)",
  "publisher": "editorial (Los Libros de El Nacional, Editorial Dahbar, etc.)",
  "isbn": "ISBN-13 si existe (solo números)",
  "subjects": ["tema1", "tema2", "tema3"]
}}

REGLAS CRÍTICAS:
- main_author: el AUTOR LITERARIO (quien escribió el texto), NO editores/coordinadores/fotógrafos
- Si solo ves "editor", "coordinador", "compilador", "fotógrafo", deja main_author vacío ""
- Excluye de title: Paper Back, Final, Tripa, KDP, Print, Texto

Texto del libro:
{full_text[:6000]}

Responde SOLO con JSON, sin explicaciones."""

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            # Parsear respuesta
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            extracted = json.loads(response_text.strip())
            
            # Actualizar metadata (solo campos no vacíos)
            for key, value in extracted.items():
                if value and value != "":
                    metadata[filename][key] = value
                    if key == 'main_author':
                        print(f"  ✅ Autor: {value}")
                    elif key == 'title':
                        print(f"  ✅ Título: {value}")
                    elif key == 'isbn':
                        print(f"  ✅ ISBN: {value}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Guardar metadata actualizada
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Proceso completado. Metadata actualizada en: {METADATA_FILE}")


if __name__ == "__main__":
    extract_metadata_for_missing_authors()
