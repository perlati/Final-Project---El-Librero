import json
from pathlib import Path

meta_path = Path("data/books_metadata_llm.json")
metadata = json.loads(meta_path.read_text(encoding="utf-8"))

updates = {
    "2013_Democracia, paz y desarrollo_Final.pdf": {
        "title": "Democracia, paz y desarrollo",
        "main_author": "",
        "editors": ["Edmundo González Urrutia"],
        "year": "2012",
        "publisher": "Editorial Melvin",
        "isbn": "9789807212151",
    },
    "2024_Texto_Democracia, paz y desarrollo_PB.pdf": {
        "title": "Democracia, paz y desarrollo",
        "main_author": "",
        "editors": ["Edmundo González Urrutia"],
        "year": "2012",
        "publisher": "Editorial Melvin",
        "isbn": "9789807212151",
    },
    "2015_Rastrilladores_de_estiercol_Paper_Back.pdf": {
        "title": "Rastrilladores de estiércol",
        "main_author": "IPYS Venezuela",
        "year": "2015",
        "publisher": "Editorial Dahbar",
        "isbn": "9789807212878",
    },
    "2016_Periodismo_al_limite_Final.pdf": {
        "title": "Periodismo al límite",
        "main_author": "",
        "editors": ["Froilán Escobar", "Ernesto Rivera"],
        "year": "2008",
        "publisher": "Fundación Educativa San Judas Tadeo",
    },
    "2021_Ahora_van_a_conocer_al_Diablo_Final.pdf": {
        "title": "Ahora van a conocer al diablo",
        "main_author": "",
        "editors": ["Oscar Medina"],
        "year": "2021",
        "publisher": "Editorial Dahbar",
        "isbn": "9789804250705",
    },
    "2024_Zigzagueando_hacia_la_democracia.pdf": {
        "title": "Zigzagueando hacia la democracia",
        "subtitle": "Cuando las dictaduras hacen elecciones",
        "main_author": "",
        "editors": ["Isabella Sanfuentes", "Sebastián Horesok"],
        "year": "2024",
        "publisher": "Editorial Dahbar",
        "isbn": "9789804251207",
    },
}

for filename, patch in updates.items():
    if filename in metadata:
        metadata[filename].update(patch)

# sincroniza entradas renombradas por ISBN o título
for key, record in list(metadata.items()):
    if not key.startswith("97"):
        continue
    key_isbn = record.get("isbn", "")
    key_title = (record.get("title") or "").strip().lower()

    for patch in updates.values():
        patch_isbn = patch.get("isbn", "")
        patch_title = (patch.get("title") or "").strip().lower()
        if (patch_isbn and patch_isbn == key_isbn) or (patch_title and patch_title == key_title):
            merged = dict(record)
            for field, value in patch.items():
                if value not in (None, ""):
                    merged[field] = value
            metadata[key] = merged
            break

meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("updated", len(updates))
