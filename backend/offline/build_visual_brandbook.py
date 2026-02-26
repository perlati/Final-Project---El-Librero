from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from backend.config import PROJECT_ROOT

MANIFEST_PATH = Path(PROJECT_ROOT) / "data" / "website_media_curated" / "curation_manifest.json"
OUTPUT_JSON = Path(PROJECT_ROOT) / "data" / "website_media_curated" / "visual_brandbook.json"
OUTPUT_MD = Path(PROJECT_ROOT) / "data" / "website_media_curated" / "visual_brandbook.md"


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-záéíóúüñ0-9]+", text.lower()) if len(tok) > 2]


def _ratio_bucket(width: int, height: int) -> str:
    if height == 0:
        return "unknown"
    ratio = width / height
    if ratio < 0.75:
        return "vertical"
    if ratio > 1.35:
        return "horizontal"
    return "cuadrada"


def _dominant_colors(image_path: Path, k: int = 5, max_side: int = 220) -> List[str]:
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            pixels = np.asarray(image).reshape(-1, 3)
    except Exception:
        return []

    if pixels.shape[0] == 0:
        return []

    # Lightweight quantization to avoid heavy clustering deps.
    quant = (pixels // 32) * 32
    unique, counts = np.unique(quant, axis=0, return_counts=True)
    top_indices = np.argsort(counts)[::-1][:k]

    colors: List[str] = []
    for idx in top_indices:
        red, green, blue = [int(v) for v in unique[idx].tolist()]
        colors.append(f"#{red:02x}{green:02x}{blue:02x}")
    return colors


def _collect_stats(manifest: Dict[str, Any]) -> Dict[str, Any]:
    clusters = manifest.get("clusters", [])

    category_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "avg_width": 0.0,
            "avg_height": 0.0,
            "avg_area": 0.0,
            "layout_buckets": Counter(),
            "year_counts": Counter(),
            "top_name_tokens": Counter(),
            "dominant_colors": Counter(),
        }
    )

    for cluster in clusters:
        category = cluster.get("category", "web_media")
        canonical = cluster.get("canonical", {})

        width = int(canonical.get("width") or 0)
        height = int(canonical.get("height") or 0)
        file_size = int(canonical.get("file_size") or 0)
        rel_path = str(canonical.get("rel_path") or "")
        year = str(canonical.get("year") or "")

        stats = category_stats[category]
        stats["count"] += 1
        stats["avg_width"] += width
        stats["avg_height"] += height
        stats["avg_area"] += max(1, width * height)
        stats["layout_buckets"][_ratio_bucket(width, height)] += 1
        if year:
            stats["year_counts"][year] += 1

        stem = Path(rel_path).stem
        for token in _tokenize(stem):
            stats["top_name_tokens"][token] += 1

        curated_rel = canonical.get("curated_path")
        if curated_rel:
            image_path = Path(PROJECT_ROOT) / curated_rel
            for color_hex in _dominant_colors(image_path):
                stats["dominant_colors"][color_hex] += 1

        _ = file_size  # keeps a hook if we want to include file-size stats later

    result: Dict[str, Any] = {"categories": {}}
    for category, stats in category_stats.items():
        count = max(1, stats["count"])
        result["categories"][category] = {
            "count": stats["count"],
            "avg_width": round(stats["avg_width"] / count, 2),
            "avg_height": round(stats["avg_height"] / count, 2),
            "avg_area": round(stats["avg_area"] / count, 2),
            "layout_distribution": dict(stats["layout_buckets"].most_common()),
            "top_years": stats["year_counts"].most_common(8),
            "top_name_tokens": stats["top_name_tokens"].most_common(25),
            "palette_top": stats["dominant_colors"].most_common(12),
        }

    return result


def _build_brand_principles(stats: Dict[str, Any]) -> List[str]:
    categories = stats.get("categories", {})
    cover = categories.get("cover", {})
    branding = categories.get("branding", {})

    cover_layouts = cover.get("layout_distribution", {})
    dominant_layout = "vertical"
    if cover_layouts:
        dominant_layout = max(cover_layouts.items(), key=lambda item: item[1])[0]

    cover_palette = [color for color, _ in cover.get("palette_top", [])[:5]]
    branding_tokens = [token for token, _ in branding.get("top_name_tokens", [])[:8]]

    principles = [
        f"Priorizar composiciones {dominant_layout} para tapas, con jerarquía tipográfica clara (título > autor > sello).",
        "Mantener alto contraste entre fondo y texto para legibilidad en miniatura y en tienda digital.",
        "Aplicar una paleta controlada por tapa (2–4 colores dominantes) para evitar ruido visual.",
        "Conservar una identidad editorial consistente: equilibrio entre sobriedad política/cultural y fuerza visual contemporánea.",
    ]

    if cover_palette:
        principles.append(f"Colores frecuentes observados en tapas: {', '.join(cover_palette)}.")
    if branding_tokens:
        principles.append(f"Elementos de branding recurrentes en nombres/activos: {', '.join(branding_tokens)}.")

    return principles


def _write_markdown(stats: Dict[str, Any], principles: List[str]) -> None:
    lines: List[str] = []
    lines.append("# Visual Brandbook – Editorial Dahbar")
    lines.append("")
    lines.append("Este documento se generó automáticamente desde `data/website_media_curated/curation_manifest.json`.")
    lines.append("")
    lines.append("## Principios de estilo")
    for principle in principles:
        lines.append(f"- {principle}")

    for category, payload in stats.get("categories", {}).items():
        lines.append("")
        lines.append(f"## Categoría: {category}")
        lines.append(f"- Activos: {payload.get('count', 0)}")
        lines.append(f"- Resolución promedio: {payload.get('avg_width', 0)} x {payload.get('avg_height', 0)}")
        lines.append(f"- Área promedio: {payload.get('avg_area', 0)}")

        layout_distribution = payload.get("layout_distribution", {})
        if layout_distribution:
            lines.append("- Distribución de layout:")
            for key, value in layout_distribution.items():
                lines.append(f"  - {key}: {value}")

        top_years = payload.get("top_years", [])
        if top_years:
            lines.append("- Años con más activos:")
            for year, count in top_years[:5]:
                lines.append(f"  - {year}: {count}")

        colors = payload.get("palette_top", [])
        if colors:
            lines.append("- Paleta dominante:")
            for color, count in colors[:8]:
                lines.append(f"  - {color}: {count}")

        tokens = payload.get("top_name_tokens", [])
        if tokens:
            token_str = ", ".join(token for token, _ in tokens[:15])
            lines.append(f"- Tokens frecuentes: {token_str}")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    manifest = _load_manifest()
    stats = _collect_stats(manifest)
    principles = _build_brand_principles(stats)

    payload = {
        "source_manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "principles": principles,
        "stats": stats,
    }

    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(stats, principles)

    print("[DONE] Visual brandbook generated")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"MD:   {OUTPUT_MD}")


if __name__ == "__main__":
    main()
