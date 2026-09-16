# RAG 基础知识（Retrieval-Augmented Generation）

> 💡 **使用指南**：本文件偏重 RAG 的原理、架构和常见优化策略，是「第二层：理论与方法」笔记。  
> 结合实战时，建议配合：`实战.md` 中的「RAG 系统」章节，以及 `code/rag_example.py` 里的完整示例代码一起阅读。

## 🎯 一句话总结

**RAG = 检索 + 生成**

让AI在回答问题时，先从知识库中检索相关信息，再基于这些信息生成答案。

---

## 📘 什么是 RAG？

**RAG（Retrieval-Augmented Generation）** = 检索增强生成

### 核心思想

```
传统LLM:
用户问题 → LLM → 答案
（只依赖训练数据，可能过时或不准确）

RAG系统:
用户问题 → 检索相关文档 → 将文档+问题一起给LLM → 答案
（基于最新、准确的知识库）
```

---

## 🤔 为什么需要 RAG？

### LLM的三大问题

| 问题 | 说明 | RAG解决方案 |
|------|------|------------|
| **知识过时** | 训练数据有截止日期 | 实时检索最新文档 |
| **幻觉问题** | 编造不存在的信息 | 基于真实文档回答 |
| **领域知识不足** | 缺乏专业/私有知识 | 接入企业知识库 |

### 示例对比

**❌ 纯LLM（可能出错）**
```
Q: 我们公司2024年Q3的营收是多少？
A: 我无法获取实时数据... [或者编造数据]
```

**✅ RAG系统（准确）**
```
Q: 我们公司2024年Q3的营收是多少？
[检索] → 找到财报文档
A: 根据2024年Q3财报，营收为XX亿元。
```

---

## 🏗️ RAG 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    RAG 系统流程                          │
└─────────────────────────────────────────────────────────┘

第一步：离线准备（只做一次）
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 原始文档  │ → │ 文本分块  │ → │ 向量化   │
│ (PDF/TXT)│    │ (Chunking)│    │(Embedding)│
└──────────┘    └──────────┘    └──────────┘
                                      ↓
                              ┌──────────┐
                              │ 向量数据库│
                              │(Vector DB)│
                              └──────────┘

第二步：在线查询（每次问答）
┌──────────┐    ┌──────────┐    ┌──────────┐
│ 用户问题  │ → │ 向量化   │ → │ 相似度搜索│
└──────────┘    │(Embedding)│    │(Retrieve) │
                └──────────┘    └──────────┘
                                      ↓
                              ┌──────────┐
                              │ Top-K文档 │
                              └──────────┘
                                      ↓
                              ┌──────────┐
                              │ 问题+文档 │ → LLM → 答案
                              │(Prompt)   │
                              └──────────┘
```

---

## 🧩 RAG 核心组件

### 1️⃣ 文档处理（Document Processing）

**文本分块（Chunking）**
```python
# 为什么要分块？
原因：
- 文档太长，超过LLM上下文限制
- 提高检索精度（小块更精准）

常见策略：
- 固定长度：每500字一块
- 语义分割：按段落/章节
- 滑动窗口：有重叠的块

示例：
原文档 (10000字)
    ↓ 分块
块1: 0-500字
块2: 500-1000字
块3: 1000-1500字
...
```

### 2️⃣ 向量化（Embedding）

**将文本转换为数字向量**
```python
文本 → Embedding模型 → 向量

示例：
"苹果很好吃" → [0.23, -0.45, 0.67, ..., 0.12]
                 (768维向量)

特点：
- 语义相似的文本，向量也相似
- 可以用数学方法计算相似度
```

**常用Embedding模型**
- OpenAI: `text-embedding-ada-002`
- 开源: `all-MiniLM-L6-v2`
- 中文: `text2vec-base-chinese`

### 3️⃣ 向量数据库（Vector Database）

**存储和检索向量**
```
传统数据库：
SELECT * FROM docs WHERE title='XXX'

向量数据库：
找到与query向量最相似的Top-K个文档
```

**常用向量数据库**
| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| **Chroma** | 轻量级，易上手 | 开发测试 |
| **Pinecone** | 云服务，托管 | 生产环境 |
| **Milvus** | 开源，可扩展 | 大规模部署 |
| **FAISS** | Facebook开源 | 本地快速检索 |
| **Weaviate** | 功能丰富 | 企业应用 |

### 4️⃣ 检索器（Retriever）

**找到最相关的文档**
```python
相似度计算：
- 余弦相似度 (最常用)
- 欧氏距离
- 点积

Top-K 选择：
- 通常取 3-5 个最相关文档
- 太少：信息不足
- 太多：噪音增加
```

### 5️⃣ 生成器（Generator）

**LLM生成最终答案**
```python
Prompt模板：
"""
基于以下文档回答问题：

文档1: ...
文档2: ...
文档3: ...

问题：{用户问题}

请基于上述文档回答，如果文档中没有相关信息，请明确说明。
"""
```

---

## 💻 完整代码示例

### 最简单的RAG实现

```python
# 安装依赖
# pip install langchain chromadb openai

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ============================================================
# 步骤1：加载文档
# ============================================================
loader = TextLoader("company_docs.txt", encoding="utf-8")
documents = loader.load()

# ============================================================
# 步骤2：文本分块
# ============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块500字符
    chunk_overlap=50     # 重叠50字符
)
chunks = text_splitter.split_documents(documents)

print(f"文档被分为 {len(chunks)} 块")

# ============================================================
# 步骤3：向量化 + 存入向量数据库
# ============================================================
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化存储
)

# ============================================================
# 步骤4：创建RAG链
# ============================================================
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 简单策略：把所有文档塞进prompt
    retriever=vectordb.as_retriever(
        search_kwargs={"k": 3}  # 检索Top-3文档
    ),
    return_source_documents=True  # 返回来源文档
)

# ============================================================
# 步骤5：使用RAG系统
# ============================================================
query = "公司2024年的战略目标是什么？"
result = qa_chain({"query": query})

print("问题:", query)
print("\n答案:", result["result"])
print("\n来源文档:")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n文档{i}:")
    print(doc.page_content[:200])  # 显示前200字符
```

### 从零实现简易RAG

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SimpleRAG:
    """简易RAG系统"""
    
    def __init__(self):
        # 使用开源Embedding模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, docs):
        """添加文档到知识库"""
        self.documents = docs
        # 向量化所有文档
        self.embeddings = self.model.encode(docs)
        print(f"✅ 已添加 {len(docs)} 个文档")
    
    def retrieve(self, query, top_k=3):
        """检索最相关的文档"""
        # 向量化查询
        query_embedding = self.model.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(
            query_embedding, 
            self.embeddings
        )[0]
        
        # 获取Top-K索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 返回文档和分数
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': similarities[idx]
            })
        
        return results
    
    def answer(self, query, llm_function):
        """生成答案"""
        # 1. 检索相关文档
        retrieved_docs = self.retrieve(query, top_k=3)
        
        # 2. 构建Prompt
        context = "\n\n".join([
            f"文档{i+1}: {doc['document']}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        prompt = f"""基于以下文档回答问题：

{context}

问题：{query}

请基于上述文档回答。如果文档中没有相关信息，请说"根据提供的文档无法回答"。
"""
        
        # 3. 调用LLM生成答案
        answer = llm_function(prompt)
        
        return {
            'answer': answer,
            'sources': retrieved_docs
        }


# 使用示例
if __name__ == "__main__":
    # 创建RAG系统
    rag = SimpleRAG()
    
    # 添加知识库
    documents = [
        "我们公司成立于2020年，主要业务是AI技术研发。",
        "2024年Q3营收达到5000万元，同比增长50%。",
        "公司总部位于北京，在上海、深圳设有分公司。",
        "我们的主要产品包括智能客服系统和数据分析平台。",
    ]
    rag.add_documents(documents)
    
    # 检索测试
    query = "公司2024年的营收情况"
    results = rag.retrieve(query)
    
    print(f"\n问题: {query}\n")
    print("检索结果:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 相似度: {result['score']:.4f}")
        print(f"   内容: {result['document']}")
```

---

## 🎯 RAG 的优化技巧

### 1️⃣ 文档处理优化

**智能分块**
```python
# ❌ 简单分块（可能切断语义）
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# ✅ 语义分块（保持完整性）
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
```

**添加元数据**
```python
# 给每个文档块添加元数据
chunk = {
    'content': "文档内容...",
    'metadata': {
        'source': 'report_2024_Q3.pdf',
        'page': 5,
        'section': '财务数据',
        'date': '2024-10-01'
    }
}
```

### 2️⃣ 检索优化

**混合检索（Hybrid Search）**
```python
# 结合向量检索和关键词检索
vector_results = vector_search(query, top_k=5)
keyword_results = bm25_search(query, top_k=5)

# 融合结果（Reciprocal Rank Fusion）
final_results = merge_results(vector_results, keyword_results)
```

**重排序（Re-ranking）**
```python
# 1. 粗检索：向量搜索Top-20
candidates = vector_search(query, top_k=20)

# 2. 精排序：使用专门的排序模型
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 计算query和每个候选的相关性分数
scores = reranker.predict([(query, doc) for doc in candidates])

# 3. 取Top-3
top_docs = [candidates[i] for i in np.argsort(scores)[-3:][::-1]]
```

### 3️⃣ 生成优化

**多策略生成**
```python
# Stuff: 把所有文档塞进一个prompt（简单但可能超长）
# Map-Reduce: 分别总结每个文档，再合并（适合长文档）
# Refine: 逐个文档迭代改进答案（更精确）

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # 或 "stuff", "refine"
    retriever=retriever
)
```

**带引用的回答**
```python
prompt = """基于以下文档回答问题，并标注信息来源：

文档1: ...
文档2: ...

问题：{query}

请在答案中标注每条信息的来源，格式：[文档X]
"""
```

---

## 📊 RAG vs 其他方案

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **纯LLM** | 简单快速 | 知识过时、幻觉 | 通用对话 |
| **RAG** | 知识可更新、准确 | 依赖文档质量 | 知识问答 |
| **微调（Fine-tuning）** | 深度定制 | 成本高、不灵活 | 特定任务 |
| **Prompt工程** | 零成本 | 效果有限 | 简单任务 |

**组合使用**
```
RAG + 微调：先RAG检索，用微调的模型生成
RAG + Prompt：优化检索prompt和生成prompt
```

---

## 🚀 实际应用场景

### 1. 企业知识库问答
```
场景：员工查询公司政策、流程
文档：规章制度、操作手册、FAQ
效果：24/7自助查询，减少HR工作量
```

### 2. 客户服务
```
场景：客户咨询产品问题
文档：产品说明书、常见问题、售后政策
效果：快速准确回答，提升客户满意度
```

### 3. 法律/医疗咨询
```
场景：专业领域问答
文档：法律条文、医学文献
效果：提供专业、有依据的建议
```

### 4. 代码助手
```
场景：开发者查询API用法
文档：代码文档、示例代码
效果：快速找到正确用法
```

### 5. 研究助手
```
场景：文献检索和总结
文档：学术论文、研究报告
效果：快速了解领域知识
```

---

## 🔧 常用工具和框架

### Python框架

**LangChain（最流行）**
```python
from langchain.document_loaders import PDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 完整的RAG工具链
```

**LlamaIndex（原GPT Index）**
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# 专注于文档索引和查询
```

**Haystack**
```python
from haystack import Pipeline
from haystack.nodes import EmbeddingRetriever

# 企业级RAG框架
```

### 向量数据库

```python
# Chroma（本地开发）
from chromadb import Client

# Pinecone（云服务）
import pinecone

# Milvus（可扩展）
from pymilvus import connections
```

---

## 🤔 常见问题

### Q1: RAG和微调（Fine-tuning）哪个好？

**A:**
```
RAG优势：
✅ 知识可实时更新
✅ 成本低（不需要重新训练）
✅ 可解释性强（可追溯来源）

微调优势：
✅ 深度定制行为
✅ 不依赖外部文档
✅ 推理速度快

选择：
- 知识密集型任务 → RAG
- 特定风格/任务 → 微调
- 最佳方案：RAG + 微调
```

### Q2: 如何选择chunk_size？

**A:**
```
小chunk (200-500字):
✅ 检索更精准
❌ 可能丢失上下文

大chunk (1000-2000字):
✅ 保留更多上下文
❌ 检索不够精确

建议：
- 技术文档：500字
- 长文小说：1000字
- FAQ：按问题分块
- 实验找最佳值
```

### Q3: 向量数据库必须用吗？

**A:**
```
小规模（<10000文档）：
- 可以用FAISS在内存中
- 甚至用NumPy暴力搜索

大规模（>10000文档）：
- 必须用专业向量数据库
- 支持分布式、增量更新
- 提供更多功能（过滤、混合搜索）
```

### Q4: 如何处理多语言文档？

**A:**
```python
# 方案1：多语言Embedding模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 方案2：翻译成统一语言
from googletrans import Translator
translator = Translator()

# 方案3：分别建索引
chinese_db = Chroma(collection_name="chinese_docs")
english_db = Chroma(collection_name="english_docs")
```