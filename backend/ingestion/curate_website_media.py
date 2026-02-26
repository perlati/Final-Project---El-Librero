from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from backend.config import PROJECT_ROOT
from backend.utils.text import normalize_title

MEDIA_ROOT = Path(PROJECT_ROOT) / "data" / "website_media" / "MEDIA"
METADATA_PATH = Path(PROJECT_ROOT) / "data" / "books_metadata_llm.json"
CURATED_ROOT = Path(PROJECT_ROOT) / "data" / "website_media_curated"
MANIFEST_PATH = Path(PROJECT_ROOT) / "data" / "website_media_curated" / "curation_manifest.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

GENERIC_FAMILY_KEYS = {
    "image",
    "img",
    "photo",
    "foto",
    "banner",
    "logo",
    "portada",
    "book",
}

STOPWORDS = {
    "de", "del", "la", "las", "el", "los", "y", "en", "a", "un", "una", "por", "para", "con", "sin",
}

COVER_KEYWORDS = {
    "cover", "portada", "tapa", "contraportada", "book", "libro", "caratula",
}

BRANDING_KEYWORDS = {
    "logo", "branding", "brand", "editorial", "dahbar", "isotipo", "imagotipo", "favicon", "banner", "header", "footer",
}


@dataclass
class ImageRecord:
    path: Path
    rel_path: str
    file_name: str
    stem: str
    base_key: str
    sha1: str
    file_size: int
    width: int
    height: int
    area: int
    aspect_ratio: float
    dhash: int
    year: str


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        if self.rank[pa] < self.rank[pb]:
            self.parent[pa] = pb
        elif self.rank[pa] > self.rank[pb]:
            self.parent[pb] = pa
        else:
            self.parent[pb] = pa
            self.rank[pa] += 1


def _tokenize(value: str) -> List[str]:
    tokens = re.findall(r"[a-záéíóúüñ0-9]+", value.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _load_title_token_sets() -> List[set[str]]:
    if not METADATA_PATH.exists():
        return []
    try:
        raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    items: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict):
        items = [item for item in raw.values() if isinstance(item, dict)]

    token_sets: List[set[str]] = []
    for item in items:
        normalized_title = normalize_title(item.get("normalized_title") or item.get("title", ""))
        tokens = set(_tokenize(normalized_title))
        if len(tokens) >= 2:
            token_sets.append(tokens)
    return token_sets


def _base_key(stem: str) -> str:
    cleaned = stem.lower()
    cleaned = re.sub(r"[-_](\d{2,5})x(\d{2,5})$", "", cleaned)
    cleaned = re.sub(r"-scaled$", "", cleaned)
    cleaned = re.sub(r"[-_](copy|copia|final|edited|editado|small|medium|large)$", "", cleaned)
    cleaned = re.sub(r"[-_](\d{1,2})$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _family_key(stem: str) -> str:
    """
    Canonical key used to merge same-name variants aggressively.

    It removes common CMS/WordPress suffixes and trailing counters,
    so files like:
      libro.jpg, libro-150x150.jpg, libro-1.webp, libro-scaled.png
    map to the same family key.
    """
    key = stem.lower().strip()
    key = re.sub(r"[-_](\d{2,5})x(\d{2,5})$", "", key)
    key = re.sub(r"[-_](scaled|copy|copia|final|edited|editado|small|medium|large)$", "", key)
    key = re.sub(r"[-_](\d{1,2})$", "", key)
    key = re.sub(r"\s+", " ", key)
    return key.strip(" -_")


def _is_generic_family_key(key: str) -> bool:
    if not key:
        return True
    if key in GENERIC_FAMILY_KEYS:
        return True
    alnum = re.sub(r"[^a-z0-9]", "", key)
    if len(alnum) <= 4:
        return True
    return False


def _dhash_64(img: Image.Image) -> int:
    resized = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.uint8)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    flat = diff.flatten()
    for bit in flat:
        bits = (bits << 1) | int(bit)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _compute_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _scan_images() -> List[ImageRecord]:
    if not MEDIA_ROOT.exists():
        raise FileNotFoundError(f"Media root not found: {MEDIA_ROOT}")

    records: List[ImageRecord] = []
    for path in MEDIA_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                dh = _dhash_64(img)
        except Exception:
            continue

        if width <= 0 or height <= 0:
            continue

        rel_path = path.relative_to(MEDIA_ROOT).as_posix()
        parts = path.parts
        year = ""
        for token in parts:
            if re.fullmatch(r"20\d{2}|19\d{2}", token):
                year = token
                break

        size = path.stat().st_size
        area = width * height
        ratio = round(width / height, 4) if height else 0.0

        records.append(
            ImageRecord(
                path=path,
                rel_path=rel_path,
                file_name=path.name,
                stem=path.stem,
                base_key=_base_key(path.stem),
                sha1=_compute_sha1(path),
                file_size=size,
                width=width,
                height=height,
                area=area,
                aspect_ratio=ratio,
                dhash=dh,
                year=year,
            )
        )

    return records


def _classify(record: ImageRecord, title_token_sets: List[set[str]]) -> Tuple[str, float, str]:
    haystack = f"{record.rel_path} {record.stem}".lower()

    if any(keyword in haystack for keyword in BRANDING_KEYWORDS):
        return "branding", 0.95, "keyword_branding"

    if any(keyword in haystack for keyword in COVER_KEYWORDS):
        return "cover", 0.9, "keyword_cover"

    stem_tokens = set(_tokenize(record.stem))
    best_overlap = 0
    best_ratio = 0.0
    for title_tokens in title_token_sets:
        overlap = len(stem_tokens & title_tokens)
        if overlap <= 0:
            continue
        ratio = overlap / max(1, len(title_tokens))
        if overlap > best_overlap or (overlap == best_overlap and ratio > best_ratio):
            best_overlap = overlap
            best_ratio = ratio

    if best_overlap >= 2 and best_ratio >= 0.4:
        return "cover", 0.85, "title_token_match"

    # portrait images with good resolution often are book covers from CMS variants
    if record.height > record.width and 1.3 <= (record.height / max(1, record.width)) <= 1.9 and record.height >= 500:
        return "cover", 0.6, "portrait_heuristic"

    return "web_media", 0.7, "default"


def _score_quality(record: ImageRecord) -> Tuple[int, int, int, int]:
    format_bonus = {".png": 2, ".webp": 1, ".jpg": 0, ".jpeg": 0, ".gif": -1}
    bonus = format_bonus.get(record.path.suffix.lower(), 0)
    return (record.area, record.file_size, bonus, record.width)


def _cluster_duplicates(records: List[ImageRecord], max_hamming: int = 8) -> Dict[int, List[int]]:
    uf = UnionFind(len(records))

    # Exact duplicate pass (binary identical)
    by_sha1: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        by_sha1.setdefault(rec.sha1, []).append(i)
    for idxs in by_sha1.values():
        root = idxs[0]
        for j in idxs[1:]:
            uf.union(root, j)

    # Force-merge same filename family (cropped/format variants)
    by_family: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        fam = _family_key(rec.stem)
        by_family.setdefault(fam, []).append(i)

    for fam, idxs in by_family.items():
        if len(idxs) <= 1:
            continue
        if _is_generic_family_key(fam):
            # Generic names are handled by visual checks below.
            continue
        root = idxs[0]
        for j in idxs[1:]:
            uf.union(root, j)

    # Near-duplicate pass constrained by base key
    by_base: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        by_base.setdefault(rec.base_key, []).append(i)

    for idxs in by_base.values():
        if len(idxs) <= 1:
            continue
        for a_pos in range(len(idxs)):
            i = idxs[a_pos]
            ri = records[i]
            for b_pos in range(a_pos + 1, len(idxs)):
                j = idxs[b_pos]
                rj = records[j]

                ratio_delta = abs((ri.width / max(1, ri.height)) - (rj.width / max(1, rj.height)))
                if ratio_delta > 0.12:
                    continue

                if _hamming(ri.dhash, rj.dhash) <= max_hamming:
                    uf.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(records)):
        parent = uf.find(i)
        clusters.setdefault(parent, []).append(i)

    return clusters


def _prepare_output_dirs() -> None:
    (CURATED_ROOT / "cover").mkdir(parents=True, exist_ok=True)
    (CURATED_ROOT / "web_media").mkdir(parents=True, exist_ok=True)
    (CURATED_ROOT / "branding").mkdir(parents=True, exist_ok=True)


def curate() -> Dict[str, Any]:
    print("[1/5] Scanning images...")
    records = _scan_images()
    print(f"  Found {len(records)} valid images")

    print("[2/5] Building title token index...")
    title_token_sets = _load_title_token_sets()

    print("[3/5] Clustering duplicates...")
    clusters = _cluster_duplicates(records)
    print(f"  Built {len(clusters)} visual groups")

    print("[4/5] Selecting best quality and classifying...")
    _prepare_output_dirs()

    manifest_clusters: List[Dict[str, Any]] = []
    category_counts = {"cover": 0, "web_media": 0, "branding": 0}
    duplicate_members = 0

    for cluster_id, idxs in enumerate(clusters.values(), start=1):
        members = [records[i] for i in idxs]
        members.sort(key=_score_quality, reverse=True)
        canonical = members[0]

        category, confidence, reason = _classify(canonical, title_token_sets)
        category_counts[category] += 1
        duplicate_members += max(0, len(members) - 1)

        target_name = f"{cluster_id:05d}__{canonical.file_name}"
        target_path = CURATED_ROOT / category / target_name
        if not target_path.exists():
            shutil.copy2(canonical.path, target_path)

        manifest_clusters.append(
            {
                "cluster_id": cluster_id,
                "category": category,
                "confidence": round(confidence, 3),
                "reason": reason,
                "canonical": {
                    "rel_path": canonical.rel_path,
                    "curated_path": target_path.relative_to(PROJECT_ROOT).as_posix(),
                    "width": canonical.width,
                    "height": canonical.height,
                    "file_size": canonical.file_size,
                    "sha1": canonical.sha1,
                    "year": canonical.year,
                },
                "members": [
                    {
                        "rel_path": rec.rel_path,
                        "width": rec.width,
                        "height": rec.height,
                        "file_size": rec.file_size,
                        "sha1": rec.sha1,
                    }
                    for rec in members
                ],
            }
        )

    print("[5/5] Writing manifest...")
    manifest = {
        "source_root": MEDIA_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "curated_root": CURATED_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "total_images_scanned": len(records),
        "unique_visual_assets": len(manifest_clusters),
        "duplicate_members_removed": duplicate_members,
        "category_counts": category_counts,
        "clusters": manifest_clusters,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest


def main() -> None:
    manifest = curate()
    print("\n=== DONE ===")
    print(f"Total images scanned: {manifest['total_images_scanned']}")
    print(f"Unique visual assets: {manifest['unique_visual_assets']}")
    print(f"Duplicates removed: {manifest['duplicate_members_removed']}")
    print(f"Category counts: {manifest['category_counts']}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
