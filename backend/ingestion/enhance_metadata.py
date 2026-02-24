#!/usr/bin/env python3
"""
Script para enriquecer metadata de libros:
- Extrae ISBN de PDFs
- Busca información en editorialdahbar.com
- Extrae años de nombres de archivos
- Profundiza extracción de metadata
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
import openai
from dotenv import load_dotenv

load_dotenv()

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent.parent
BOOKS_DIR = PROJECT_ROOT / "data" / "books"
METADATA_FILE = PROJECT_ROOT / "data" / "books_metadata_llm.json"

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"


def extract_year_from_filename(filename: str) -> Optional[str]:
    """Extrae el año del nombre de archivo (formato: YYYY_Titulo...)"""
    match = re.match(r'^(\d{4})_', filename)
    return match.group(1) if match else None


def extract_isbn_from_text(text: str) -> Optional[str]:
    """Busca patrones de ISBN-10 o ISBN-13 en texto"""
    # ISBN-13: 978-X-XXXX-XXXX-X o 9789801234567
    isbn13_pattern = r'(?:ISBN(?:-13)?:?\s*)?(?:978)?[-\s]?(\d{1,5})[-\s]?(\d{1,7})[-\s]?(\d{1,7})[-\s]?(\d{1})'
    match13 = re.search(isbn13_pattern, text, re.IGNORECASE)
    
    if match13:
        # Limpiar y retornar solo dígitos
        isbn = ''.join(match13.groups())
        if len(isbn) == 10:
            isbn = '978' + isbn  # Convertir ISBN-10 a 13
        if len(isbn) == 13:
            return isbn
    
    # ISBN-10: X-XXXX-XXXX-X
    isbn10_pattern = r'(?:ISBN(?:-10)?:?\s*)?(\d{1,5})[-\s]?(\d{1,7})[-\s]?(\d{1,7})[-\s]?(\d{1})'
    match10 = re.search(isbn10_pattern, text, re.IGNORECASE)
    
    if match10:
        isbn = ''.join(match10.groups())
        if len(isbn) == 10:
            return '978' + isbn
    
    return None


def extract_isbn_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extrae ISBN de las primeras 10 páginas de un PDF"""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        num_pages = min(10, len(pages))
        for i in range(num_pages):
            text = pages[i].page_content
            isbn = extract_isbn_from_text(text)
            if isbn:
                return isbn
    except Exception as e:
        print(f"Error extrayendo ISBN de {pdf_path.name}: {e}")
    
    return None


def search_editorial_website(title: str, author: str) -> Dict[str, Any]:
    """Busca información del libro en editorialdahbar.com"""
    try:
        # Intentar búsqueda en catálogo
        search_url = f"https://editorialdahbar.com/catalogo?search={title}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Buscar información relevante (adaptar según estructura real del sitio)
            # Esto es un placeholder que deberás ajustar según la estructura real
            return {"found": True, "source": "web"}
    except Exception as e:
        print(f"Error buscando en web para '{title}': {e}")
    
    return {"found": False}


def enhance_metadata_with_llm(pdf_path: Path, current_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Usa LLM para extraer metadata más profunda del PDF"""
    try:
        # Leer más páginas (primeras 15 en lugar de 6)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        num_pages = min(15, len(pages))
        text_chunks = []
        for i in range(num_pages):
            text = pages[i].page_content
            text_chunks.append(f"--- Página {i + 1} ---\n{text[:2000]}")
        
        full_text = "\n\n".join(text_chunks)
        
        # LLM para extraer metadata mejorada
        prompt = f"""Analiza las primeras páginas de este libro y extrae la siguiente información en formato JSON:

{{
  "title": "título completo del libro",
  "subtitle": "subtítulo si existe",
  "main_author": "autor principal",
  "other_authors": ["otros autores si existen"],
  "year": "año de publicación",
  "publisher": "editorial",
  "isbn": "ISBN-13 si lo encuentras (solo números)",
  "subjects": ["tema1", "tema2", "tema3"],
  "language": "código de idioma (es/en)",
  "page_count": "número total de páginas si lo menciona"
}}

IMPORTANTE:
- Si no encuentras un campo, usa una cadena vacía "" o lista vacía []
- Para subjects, identifica 3-5 temas principales del libro
- Busca el ISBN en las primeras páginas (formato 978-XXXXXXXXXX)
- Si ves "Los Libros de El Nacional", ese es el publisher
- Si ves "Editorial Dahbar", ese es el publisher

Texto del libro:
{full_text[:8000]}

Responde SOLO con el JSON, sin explicaciones adicionales."""

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Parsear respuesta
        response_text = response.choices[0].message.content.strip()
        
        # Limpiar markdown si existe
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        enhanced = json.loads(response_text.strip())
        
        # Mergear con metadata existente (priorizar lo nuevo si no está vacío)
        for key, value in enhanced.items():
            if value and (not current_metadata.get(key) or current_metadata.get(key) in ["", [], "Desconocido"]):
                current_metadata[key] = value
        
        return current_metadata
        
    except Exception as e:
        print(f"Error mejorando metadata con LLM para {pdf_path.name}: {e}")
        return current_metadata


def enhance_all_metadata():
    """Proceso principal de enriquecimiento de metadata"""
    
    # Cargar metadata existente
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"📚 Procesando {len(metadata)} libros...\n")
    
    enhanced_count = 0
    isbn_found = 0
    
    for filename, meta in metadata.items():
        print(f"Procesando: {filename}")
        
        pdf_path = BOOKS_DIR / filename
        if not pdf_path.exists():
            print(f"  ⚠️  PDF no encontrado, saltando...")
            continue
        
        # 1. Extraer año del nombre de archivo si no existe
        if not meta.get("year") or meta["year"] == "":
            year = extract_year_from_filename(filename)
            if year:
                meta["year"] = year
                print(f"  ✅ Año extraído del nombre: {year}")
        
        # 2. Extraer ISBN del PDF si no existe
        if not meta.get("isbn") or meta["isbn"] == "":
            isbn = extract_isbn_from_pdf(pdf_path)
            if isbn:
                meta["isbn"] = isbn
                isbn_found += 1
                print(f"  ✅ ISBN encontrado: {isbn}")
        
        # 3. Buscar en web de Editorial (si tiene título y autor)
        if meta.get("title") and meta.get("main_author"):
            web_info = search_editorial_website(meta["title"], meta["main_author"])
            if web_info.get("found"):
                print(f"  🌐 Información encontrada en web")
        
        # 4. Mejorar metadata con LLM (solo si faltan campos importantes)
        needs_enhancement = (
            not meta.get("subjects") or 
            len(meta.get("subjects", [])) == 0 or
            not meta.get("isbn") or
            meta.get("main_author") == "Desconocido"
        )
        
        if needs_enhancement:
            print(f"  🤖 Mejorando metadata con LLM...")
            meta = enhance_metadata_with_llm(pdf_path, meta)
            enhanced_count += 1
        
        # Actualizar en el diccionario
        metadata[filename] = meta
        print()
    
    # Guardar metadata enriquecida
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Proceso completado:")
    print(f"   - {enhanced_count} libros enriquecidos con LLM")
    print(f"   - {isbn_found} ISBNs encontrados")
    print(f"   - Metadata guardada en: {METADATA_FILE}")


if __name__ == "__main__":
    enhance_all_metadata()
