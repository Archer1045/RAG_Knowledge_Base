"""骑手招聘助手 RAG 本地检索：BM25 + 区域/最近站点。"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import jieba

    jieba.setLogLevel(60)
except ImportError:  # pragma: no cover
    jieba = None

HERE = Path(__file__).resolve().parent
CHUNKS_FILE = HERE / "data" / "chunks.jsonl"
INDEX_FILE = HERE / "data" / "index.json"
EMBEDDINGS_FILE = HERE / "data" / "embeddings.npy"
VECTOR_INDEX_FILE = HERE / "data" / "vector_index.json"
MODELS_DIR = HERE / "models"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
ASCII_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
MIXED_RE = re.compile(r"\d+[\u4e00-\u9fff]|[\u4e00-\u9fff]\d+")

_CACHE: dict[str, Any] = {
    "chunks": None,
    "bm25": None,
    "vectors": None,
    "vector_meta": None,
    "embedder": None,
}


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_index(path: Path = INDEX_FILE) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if VECTOR_INDEX_FILE.exists():
        data["vector"] = json.loads(VECTOR_INDEX_FILE.read_text(encoding="utf-8"))
    return data


def _load_vectors() -> tuple[Any | None, dict[str, Any] | None]:
    if _CACHE["vectors"] is None and EMBEDDINGS_FILE.exists():
        import numpy as np

        raw = np.load(EMBEDDINGS_FILE)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _CACHE["vectors"] = (raw / norms).astype(np.float32)
        _CACHE["vector_meta"] = (
            json.loads(VECTOR_INDEX_FILE.read_text(encoding="utf-8")) if VECTOR_INDEX_FILE.exists() else None
        )
    return _CACHE["vectors"], _CACHE["vector_meta"]


def _embed_query(text: str):
    import numpy as np

    if _CACHE["embedder"] is None:
        from fastembed import TextEmbedding

        _CACHE["embedder"] = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODELS_DIR))
    vector = np.asarray(next(_CACHE["embedder"].embed([text])), dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def tokenize(text: str) -> list[str]:
    lower = str(text or "").lower()
    tokens: list[str] = []
    if jieba is not None:
        tokens.extend(jieba.cut(lower, cut_all=False))
    tokens.extend(ASCII_RE.findall(lower))
    tokens.extend(MIXED_RE.findall(lower))
    for segment in CJK_RE.findall(lower):
        if len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return [token for token in tokens if token.strip()]


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.doc_terms: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.df: Counter[str] = Counter()
        self.num_docs = len(chunks)
        for chunk in chunks:
            terms = Counter(tokenize(f"{chunk['title']}\n{chunk['content']}"))
            self.doc_terms.append(terms)
            self.doc_lengths.append(sum(terms.values()))
            self.df.update(terms.keys())
        self.avgdl = sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0
        self.k1 = 1.5
        self.b = 0.75

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1.0 + (self.num_docs - n + 0.5) / (n + 0.5))

    def score_doc(self, doc_index: int, query_terms: list[str]) -> float:
        terms = self.doc_terms[doc_index]
        length = self.doc_lengths[doc_index]
        score = 0.0
        for term in query_terms:
            tf = terms.get(term, 0)
            if not tf:
                continue
            idf_value = self.idf(term)
            score += idf_value * (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * length / self.avgdl))
        return score


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))


def get_cache() -> tuple[list[dict[str, Any]], BM25Index]:
    if _CACHE["chunks"] is None:
        chunks = load_chunks()
        _CACHE["chunks"] = chunks
        _CACHE["bm25"] = BM25Index(chunks)
    return _CACHE["chunks"], _CACHE["bm25"]


def retrieve(
    query: str = "",
    top_k: int = 5,
    district: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    type_filter: str | None = None,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    chunks, bm25 = get_cache()
    query_text = str(query or "").strip()
    query_terms = tokenize(query_text)
    has_text = bool(query_text)
    district_names = sorted({chunk["metadata"].get("district") for chunk in chunks if chunk["metadata"].get("district")})
    found_districts = [name for name in district_names if name in query_text]

    pool = list(range(len(chunks)))
    if type_filter:
        pool = [index for index in pool if chunks[index]["type"] == type_filter]
    if mode == "near":
        pool = [
            index
            for index in pool
            if chunks[index]["type"] == "site"
            and to_float(chunks[index]["metadata"].get("latitude")) is not None
            and to_float(chunks[index]["metadata"].get("longitude")) is not None
        ]
    if not has_text and district:
        pool = [
            index
            for index in pool
            if chunks[index]["metadata"].get("district") == district and chunks[index]["type"] in ("site", "district")
        ]

    vectors, vector_meta = _load_vectors()
    use_vectors = mode in ("vector", "hybrid") and vectors is not None and has_text
    if mode == "vector" and vectors is None:
        print("[warning] 未找到 embeddings.npy，回退到 BM25 文本检索", file=sys.stderr)
    query_vector = _embed_query(query_text) if use_vectors else None
    bm25_scores = {index: bm25.score_doc(index, query_terms) for index in pool}
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0
    vector_scores: dict[int, float] = {}
    if query_vector is not None and vectors is not None:
        cosines = vectors @ query_vector
        vector_scores = {index: float(cosines[index]) for index in pool}

    scored = []
    for doc_index in pool:
        chunk = chunks[doc_index]
        meta = chunk.get("metadata") or {}
        score = 0.0
        distance = None
        if mode == "near":
            pass
        elif mode == "vector" and query_vector is not None:
            score = vector_scores.get(doc_index, 0.0)
        elif mode == "hybrid" and query_vector is not None:
            bm25_norm = bm25_scores[doc_index] / max_bm25 if max_bm25 > 0 else 0.0
            score = vector_scores.get(doc_index, 0.0) + 0.25 * bm25_norm
        else:
            score = bm25.score_doc(doc_index, query_terms)

        if mode != "near":
            if district and meta.get("district") == district:
                score += 3.0
                if chunk["type"] == "district":
                    score += 5.0
            for found in found_districts:
                if meta.get("district") == found:
                    score += 3.0
                    if chunk["type"] == "district":
                        score += 5.0
            if meta.get("topic") and meta["topic"] in query_text:
                score += 2.0
            if chunk["type"] == "site" and chunk["title"] in query_text:
                score += 5.0
            if chunk["type"] == "faq" and meta.get("question") in query_text:
                score += 5.0
        site_lat = to_float(meta.get("latitude"))
        site_lon = to_float(meta.get("longitude"))
        if lat is not None and lon is not None and site_lat is not None and site_lon is not None:
            distance = haversine_km(lat, lon, site_lat, site_lon)
            if mode == "near":
                score = -distance
            else:
                score += 2.0 / (1.0 + distance)
        scored.append((score, doc_index, distance))

    scored.sort(key=lambda item: (-item[0], chunks[item[1]]["id"]))
    results = []
    for rank, (score, doc_index, distance) in enumerate(scored[:top_k], 1):
        chunk = chunks[doc_index]
        results.append(
            {
                "rank": rank,
                "id": chunk["id"],
                "source": chunk["source"],
                "type": chunk["type"],
                "title": chunk["title"],
                "score": round(score, 4),
                "distance_km": round(distance, 2) if distance is not None else None,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="骑手招聘助手 RAG 本地检索")
    parser.add_argument("query", nargs="?", default="", help="检索内容")
    parser.add_argument("--top", type=int, default=5, help="返回条数")
    parser.add_argument("--district", help="按区域优先，例如 朝阳区")
    parser.add_argument("--lat", type=float, help="纬度，用于最近站点")
    parser.add_argument("--lon", type=float, help="经度，用于最近站点")
    parser.add_argument("--type", choices=["faq", "instruction", "site", "district"], help="只检索指定类型")
    parser.add_argument("--mode", choices=["hybrid", "text", "vector", "near"], default="hybrid", help="检索模式")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--stats", action="store_true", help="输出知识库统计")
    args = parser.parse_args()

    if args.stats:
        print(INDEX_FILE.read_text(encoding="utf-8"))
        return
    if not args.query and args.lat is None and args.lon is None and not args.district:
        parser.print_help()
        sys.exit(1)

    results = retrieve(
        query=args.query,
        top_k=args.top,
        district=args.district,
        lat=args.lat,
        lon=args.lon,
        type_filter=args.type,
        mode=args.mode,
    )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if not results:
        print("未找到匹配内容")
        return
    for item in results:
        distance = f"，距离约 {item['distance_km']} 公里" if item["distance_km"] is not None else ""
        print(f"{item['rank']}. [{item['type']}] {item['title']}（score={item['score']}{distance}）")
        preview = item["content"].replace("\n", " | ")
        print(f"   {preview[:180]}")


if __name__ == "__main__":
    main()
