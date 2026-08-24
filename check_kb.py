"""知识库自检：校验生成结果并执行示例检索。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from retrieve import retrieve

HERE = Path(__file__).resolve().parent


def main() -> int:
    chunks_file = HERE / "data" / "chunks.jsonl"
    index_file = HERE / "data" / "index.json"
    chunks = [json.loads(line) for line in chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = json.loads(index_file.read_text(encoding="utf-8"))
    vectors_file = HERE / "data" / "embeddings.npy"
    vector_meta_file = HERE / "data" / "vector_index.json"
    vector_ok = False
    if vectors_file.exists() and vector_meta_file.exists():
        vectors = np.load(vectors_file)
        vector_ok = vectors.shape[0] == len(chunks) and vectors.shape[1] == 512

    expected_districts = {
        "朝阳区",
        "海淀区",
        "丰台区",
        "西城区",
        "昌平区",
        "通州区",
        "大兴区",
        "石景山区",
        "房山区",
        "顺义区",
        "东城区",
        "门头沟区",
    }
    districts = {chunk["metadata"]["district"] for chunk in chunks if chunk["type"] == "site"}
    checks = {
        "chunk_count": len(chunks) == index["counts"]["chunks"],
        "types": dict(Counter(chunk["type"] for chunk in chunks)),
        "districts_complete": districts == expected_districts,
        "first_faq_title": chunks[0]["title"] == "我可以去贵公司面试吗？",
        "all_sites_have_coords": all(
            chunk["metadata"].get("latitude") is not None and chunk["metadata"].get("longitude") is not None
            for chunk in chunks
            if chunk["type"] == "site"
        ),
        "vector_index_ok": vector_ok,
    }
    print("== 数据校验 ==")
    for key, value in checks.items():
        print(f"{key}: {value}")

    samples = [
        ("新手首月工资多少", {}),
        ("新手首月工资多少", {"mode": "vector", "top_k": 3}),
        ("朝阳区有哪些站点", {"district": "朝阳区", "top_k": 3}),
        ("附近站点", {"lat": 39.92, "lon": 116.47, "mode": "near", "top_k": 3}),
    ]
    print("== 示例检索 ==")
    for query, options in samples:
        results = retrieve(query, **options)
        print(f"\n查询: {query}")
        for item in results[:3]:
            distance = f" 距离={item['distance_km']}km" if item["distance_km"] is not None else ""
            print(f"  {item['rank']}. [{item['type']}] {item['title']} (score={item['score']}{distance})")

    passed = all(isinstance(value, bool) and value for key, value in checks.items() if key != "types")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
