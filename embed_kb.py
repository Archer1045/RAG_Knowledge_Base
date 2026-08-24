"""为知识库分块生成本地向量索引（fastembed + BAAI/bge-small-zh-v1.5）。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODELS_DIR = HERE / "models"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
VECTOR_INDEX_FILE = DATA_DIR / "vector_index.json"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    from fastembed import TextEmbedding

    chunks = load_chunks()
    if not chunks:
        sys.exit("chunks.jsonl 为空，请先运行 build_kb.py")

    print(f"加载模型 {MODEL_NAME} ...")
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODELS_DIR))
    contents = [chunk["content"] for chunk in chunks]
    vectors = np.vstack([np.asarray(vec, dtype=np.float32) for vec in model.embed(contents)])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, vectors)
    vector_index = {
        "model": MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "normalized": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    VECTOR_INDEX_FILE.write_text(json.dumps(vector_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(vector_index, ensure_ascii=False, indent=2))
    print(f"向量索引已生成: {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
