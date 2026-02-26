from pathlib import Path
import re


def normalize_title(raw: str) -> str:
    """
    Normalize book titles and filename-like strings into a clean display title.

    Examples:
    - "2018_Venezuela_en_el_nudo_gordiano_Paper_Back.pdf"
      -> "Venezuela en el nudo gordiano"
    - "Libres" -> "Libres"
    """
    if not raw:
        return ""

    name = Path(str(raw)).stem.strip()

    if len(name) > 4 and name[:4].isdigit():
        if len(name) == 4:
            name = ""
        elif name[4] in ("_", "-", " "):
            name = name[5:]

    suffixes = [
        "_Paper_Back",
        "_PaperBack",
        "_Tripa_Final",
        "_Tripa",
        "_Final",
        "_Draft",
        "_Revised",
    ]
    for suffix in suffixes:
        name = name.replace(suffix, "")

    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()
