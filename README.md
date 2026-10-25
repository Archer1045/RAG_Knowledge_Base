# 骑手招聘助手 RAG 知识库

根据 `ai-workbench` 目录下的三个源文件自动生成：

- `faq.txt`：122 条招聘问答，按“问/答”拆成独立分块
- `instructions.txt`：45 条岗位规则，按主题逐条拆成独立分块
- `sites_geocoded.xlsx`：218 个北京骑手站点，按站点逐条分块，并按行政区生成概览分块

## 目录结构

```text
rag_knowledge_base/
├── build_kb.py        # 一键从源文件生成知识库
├── embed_kb.py        # 生成本地向量索引（fastembed + bge-small-zh）
├── retrieve.py        # 本地检索（向量 + BM25 + 区域优先 + 最近站点）
├── server.py          # 本地 HTTP 检索服务
├── requirements.txt   # Python 依赖
├── README.md
├── models/            # 本地向量模型缓存
├── data/
│   ├── chunks.jsonl   # RAG 分块，可直接用于向量化/索引
│   ├── documents.json # 结构化原始数据
│   ├── index.json     # 生成统计与元信息
│   ├── embeddings.npy # 分块向量（顺序与 chunks.jsonl 一致）
│   └── vector_index.json # 向量模型与索引信息
└── documents/
    ├── faq.md
    ├── instructions.md
    └── sites.md
```

运行 `python check_kb.py` 可以快速自检数据完整性并演示检索方式。

## 使用方式

使用工作区虚拟环境：

```powershell
D:\workSpace\code\.venv\Scripts\python.exe build_kb.py
```

如果源文件不在上级目录，可以指定路径：

```powershell
python build_kb.py D:\workSpace\code\ai-workbench
```

生成本地向量索引（首次会下载约 100MB 中文向量模型）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
D:\workSpace\code\.venv\Scripts\python.exe embed_kb.py
```

## 检索示例

```powershell
# 混合检索（默认，向量 + BM25）
python retrieve.py "新手首月工资多少"

# 纯向量检索
python retrieve.py "新手首月工资多少" --mode vector

# 纯文本 BM25 检索
python retrieve.py "新手首月工资多少" --mode text

# 区域优先
python retrieve.py "朝阳区有哪些站点" --district 朝阳区

# 最近站点（传经纬度）
python retrieve.py "" --lat 39.92 --lon 116.47 --mode near --top 5

# 输出 JSON
python retrieve.py "有宿舍吗" --json
```

## 启动检索服务

```powershell
python server.py
```

默认监听 `http://localhost:8787`，接口：

- `GET /`：知识库统计
- `POST /retrieve`：检索，请求体示例

```json
{
  "query": "附近有站点吗",
  "top_k": 5,
  "lat": 39.92,
  "lon": 116.47
}
```

## 分块策略

- `faq`：每条问答一个分块，元数据包含 `question`、`answer`
- `instruction`：每条规则一个分块，元数据包含 `topic`、`value`
- `site`：每个站点一个分块，元数据包含 `district`、`address`、`latitude`、`longitude`
- `district`：每个行政区的站点概览，元数据包含 `site_count`、`site_names`

## 向量化说明

- 模型：`BAAI/bge-small-zh-v1.5`，512 维，ONNX 本地推理
- `data/embeddings.npy` 每一行对应 `data/chunks.jsonl` 中相同顺序的分块
- 检索模式：`text`（BM25）、`vector`（余弦相似度）、`hybrid`（默认，向量 + BM25）、`near`（经纬度最近站点）
- 如果未生成向量文件，`vector`/`hybrid` 模式会自动回退到 BM25，不会报错

## 接入向量数据库

`data/chunks.jsonl` 每行是一个标准分块对象：

```json
{
  "id": "faq-001",
  "source": "faq.txt",
  "type": "faq",
  "title": "问句",
  "content": "分块正文",
  "metadata": {}
}
```

可以用任意 embedding 服务把 `content` 转成向量，配合 Chroma、pgvector、FAISS 等向量库做语义检索；`metadata` 可直接作为过滤条件（例如按 `district` 过滤、按经纬度做最近站点召回）。
