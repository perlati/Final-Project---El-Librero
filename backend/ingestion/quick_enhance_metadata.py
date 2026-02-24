#!/usr/bin/env python3
"""
Script rápido para enriquecer metadata sin LLM:
- Extrae años de nombres de archivos
- Extrae ISBN de PDFs (búsqueda de patrones)
- Actualiza metadata
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent.parent
BOOKS_DIR = PROJECT_ROOT / "data" / "books"
METADATA_FILE = PROJECT_ROOT / "data" / "books_metadata_llm.json"


def extract_year_from_filename(filename: str) -> Optional[str]:
    """Extrae el año del nombre de archivo (formato: YYYY_Titulo...)"""
    match = re.match(r'^(\d{4})_', filename)
    return match.group(1) if match else None


def extract_isbn_from_text(text: str) -> Optional[str]:
    """Busca patrones de ISBN-10 o ISBN-13 en texto"""
    # ISBN-13: 978-X-XXXX-XXXX-X o 9789801234567
    patterns = [
        r'ISBN[:\s-]*?(978[\d\s-]{10,17})',  # ISBN-13 con prefijo
        r'ISBN[:\s-]*([\d\s-]{10,17})',  # ISBN genérico
        r'(978\d{10})',  # ISBN-13 sin guiones
        r'(978[\d-]{13})',  # ISBN-13 con guiones
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Limpiar: solo dígitos
            isbn = re.sub(r'[^\d]', '', match.group(1))
            if len(isbn) == 13 and isbn.startswith('978'):
                return isbn
            elif len(isbn) == 10:
                # Convertir ISBN-10 a ISBN-13
                return '978' + isbn
    
    return None


def extract_isbn_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extrae ISBN de las primeras 5 páginas de un PDF"""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        num_pages = min(5, len(pages))
        for i in range(num_pages):
            text = pages[i].page_content
            isbn = extract_isbn_from_text(text)
            if isbn:
                return isbn
    except Exception as e:
        print(f"  ⚠️  Error extrayendo ISBN: {e}")
    
    return None


def quick_enhance_metadata():
    """Enriquecimiento rápido (años + ISBN) sin LLM"""
    
    # Cargar metadata existente
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"📚 Enriqueciendo metadata de {len(metadata)} libros...\n")
    
    years_added = 0
    isbns_found = 0
    
    for filename, meta in metadata.items():
        print(f"{filename}")
        
        pdf_path = BOOKS_DIR / filename
        if not pdf_path.exists():
            print(f"  ⚠️  PDF no encontrado")
            continue
        
        # 1. Extraer año del nombre de archivo
        if not meta.get("year") or meta["year"] == "":
            year = extract_year_from_filename(filename)
            if year:
                meta["year"] = year
                years_added += 1
                print(f"  ✅ Año: {year}")
        
        # 2. Extraer ISBN del PDF (solo si no existe)
        if not meta.get("isbn") or meta["isbn"] == "":
            isbn = extract_isbn_from_pdf(pdf_path)
            if isbn:
                meta["isbn"] = isbn
                isbns_found += 1
                print(f"  ✅ ISBN: {isbn}")
        
        # Actualizar en el diccionario
        metadata[filename] = meta
    
    # Guardar metadata enriquecida
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Proceso completado:")
    print(f"   - {years_added} años agregados desde nombres de archivo")
    print(f"   - {isbns_found} ISBNs encontrados en PDFs")
    print(f"   - Metadata actualizada: {METADATA_FILE}")


if __name__ == "__main__":
    quick_enhance_metadata()
