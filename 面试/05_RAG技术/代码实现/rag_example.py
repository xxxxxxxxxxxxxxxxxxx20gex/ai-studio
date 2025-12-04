"""
RAG (Retrieval-Augmented Generation) 完整示例
从零实现一个简单但完整的RAG系统
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# 如果有安装sentence-transformers，使用真实的embedding
try:
    from sentence_transformers import SentenceTransformer
    USE_REAL_EMBEDDING = True
except ImportError:
    print("⚠️  未安装 sentence-transformers，将使用模拟embedding")
    print("安装命令: pip install sentence-transformers")
    USE_REAL_EMBEDDING = False


# ============================================================
# 简易 Embedding 实现（仅用于演示）
# ============================================================

class SimpleEmbedding:
    """简单的文本向量化（用于演示，实际应使用专业模型）"""
    
    def __init__(self, dim=100):
        self.dim = dim
        # 简单的词汇表
        self.vocab = {}
        self.next_id = 0
    
    def _get_word_id(self, word):
        """获取词的ID"""
        if word not in self.vocab:
            self.vocab[word] = self.next_id
            self.next_id += 1
        return self.vocab[word]
    
    def encode(self, texts):
        """文本转向量（简化版，仅演示用）"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # 简单的bag-of-words + 随机投影
            words = text.lower().split()
            vec = np.zeros(self.dim)
            for word in words:
                word_id = self._get_word_id(word)
                # 简单hash到向量空间
                np.random.seed(word_id)
                vec += np.random.randn(self.dim)
            
            # 归一化
            if np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        
        return np.array(embeddings)


# ============================================================
# RAG 系统核心类
# ============================================================

class SimpleRAG:
    """
    简易RAG系统
    
    功能：
    1. 添加文档到知识库
    2. 向量化和存储
    3. 检索相关文档
    4. 生成答案（模拟LLM）
    """
    
    def __init__(self, use_real_model=USE_REAL_EMBEDDING):
        """初始化RAG系统"""
        print("🚀 初始化 RAG 系统...")
        
        # 选择Embedding模型
        if use_real_model:
            print("使用真实的 Embedding 模型...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            print("使用简易 Embedding 模型（仅演示用）...")
            self.embedder = SimpleEmbedding()
        
        # 存储
        self.documents = []  # 原始文档
        self.embeddings = None  # 文档向量
        self.metadata = []  # 文档元数据
        
        print("✅ 初始化完成！\n")
    
    def add_documents(self, docs: List[str], metadata: List[Dict] = None):
        """
        添加文档到知识库
        
        参数:
            docs: 文档列表
            metadata: 每个文档的元数据（可选）
        """
        print(f"📚 添加 {len(docs)} 个文档到知识库...")
        
        # 保存文档
        self.documents.extend(docs)
        
        # 保存元数据
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(docs))
        
        # 向量化
        print("⚙️  正在向量化文档...")
        new_embeddings = self.embedder.encode(docs)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        print(f"✅ 成功添加！当前知识库共 {len(self.documents)} 个文档\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索最相关的文档
        
        参数:
            query: 查询问题
            top_k: 返回前K个最相关文档
        
        返回:
            文档列表，每个包含 content, score, metadata
        """
        if len(self.documents) == 0:
            print("⚠️  知识库为空，请先添加文档！")
            return []
        
        print(f"🔍 检索问题: '{query}'")
        
        # 向量化查询
        query_embedding = self.embedder.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 获取Top-K索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 构建结果
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'content': self.documents[idx],
                'score': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })
        
        # 显示结果
        print(f"\n📄 检索到 {len(results)} 个相关文档:\n")
        for r in results:
            print(f"  {r['rank']}. 相似度: {r['score']:.4f}")
            print(f"     内容: {r['content'][:80]}...")
            if r['metadata']:
                print(f"     元数据: {r['metadata']}")
            print()
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        基于检索到的文档生成答案（模拟LLM）
        
        实际应用中，这里应该调用真实的LLM API
        """
        # 构建上下文
        context = "\n\n".join([
            f"[文档{i+1}] {doc['content']}" for i, doc in enumerate(context_docs)])
        
        # 模拟LLM响应（实际应调用GPT/Claude等）
        answer = f"""基于检索到的 {len(context_docs)} 个文档，回答如下：

相关文档内容：
{context}

[注意：这是模拟回答。实际应用中，这里会调用真实的LLM API，
如OpenAI的GPT、Anthropic的Claude等，让它基于上述文档生成准确答案]

建议的Prompt模板：
\"\"\"
基于以下文档回答问题：

{context}

问题：{query}

请基于上述文档回答。如果文档中没有相关信息，请明确说明。
\"\"\"
"""
        return answer
    
    def query(self, question: str, top_k: int = 3, return_sources: bool = True):
        """
        完整的RAG查询流程
        
        参数:
            question: 用户问题
            top_k: 检索文档数量
            return_sources: 是否返回来源文档
        """
        print("="*70)
        print("RAG 查询流程")
        print("="*70 + "\n")
        
        # 1. 检索
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            return "知识库为空，无法回答。"
        
        # 2. 生成答案
        print("💬 生成答案...\n")
        answer = self.generate_answer(question, retrieved_docs)
        
        # 3. 返回结果
        if return_sources:
            return {
                'answer': answer,
                'sources': retrieved_docs
            }
        else:
            return answer


# ============================================================
# 演示和测试
# ============================================================

def demo_basic():
    """基础演示：简单的RAG使用"""
    print("\n" + "🎯 "*20)
    print("演示1: RAG 基础使用")
    print("🎯 "*20 + "\n")
    
    # 创建RAG系统
    rag = SimpleRAG()
    
    # 准备知识库文档
    documents = [
        "苹果公司成立于1976年，由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩创立。",
        "iPhone是苹果公司于2007年推出的智能手机产品线。",
        "苹果公司的总部位于美国加利福尼亚州库比蒂诺。",
        "MacBook是苹果公司的笔记本电脑产品系列。",
        "iOS是苹果公司为iPhone开发的移动操作系统。",
    ]
    
    # 添加文档
    rag.add_documents(documents)
    
    # 查询问题
    questions = [
        "苹果公司是什么时候成立的？",
        "iPhone是什么？",
        "苹果总部在哪里？",
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"问题 {i}: {q}")
        print('='*70)
        result = rag.query(q, top_k=2)
        print(f"\n答案:\n{result['answer']}\n")
        
        input("\n按回车继续...")


def demo_with_metadata():
    """演示2：带元数据的文档管理"""
    print("\n" + "🎯 "*20)
    print("演示2: 使用元数据")
    print("🎯 "*20 + "\n")
    
    rag = SimpleRAG()
    
    # 带元数据的文档
    documents = [
        "2024年Q1营收达到3000万元，同比增长40%。",
        "2024年Q2营收达到3500万元，环比增长16.7%。",
        "2024年Q3营收达到5000万元，创历史新高。",
    ]
    
    metadata = [
        {"source": "财报", "quarter": "Q1", "year": 2024},
        {"source": "财报", "quarter": "Q2", "year": 2024},
        {"source": "财报", "quarter": "Q3", "year": 2024},
    ]
    
    rag.add_documents(documents, metadata)
    
    # 查询
    result = rag.query("2024年Q3的营收情况", top_k=3)
    print(f"\n最终答案:\n{result['answer']}")


def demo_comparison():
    """演示3：对比检索效果"""
    print("\n" + "🎯 "*20)
    print("演示3: 检索效果对比")
    print("🎯 "*20 + "\n")
    
    rag = SimpleRAG()
    
    # 添加技术文档
    tech_docs = [
        "Python是一种高级编程语言，以其简洁和可读性著称。",
        "JavaScript是网页开发的核心技术，运行在浏览器中。",
        "Java是一种面向对象的编程语言，广泛应用于企业级应用。",
        "C++是一种性能强大的编程语言，常用于系统编程。",
        "Go语言由Google开发，专注于并发和云计算。",
    ]
    
    rag.add_documents(tech_docs)
    
    # 不同的查询
    queries = [
        "哪种语言适合网页开发？",
        "简洁易读的编程语言",
        "Google开发的语言",
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"查询: {query}")
        print('='*70)
        rag.retrieve(query, top_k=3)
        input("\n按回车继续...")


# ============================================================
# 主程序
# ============================================================

def main():
    """主函数：运行所有演示"""
    print("\n" + "📚 "*25)
    print("RAG (Retrieval-Augmented Generation) 完整演示")
    print("📚 "*25)
    
    print("\n这个程序将演示：")
    print("1. RAG的基础使用")
    print("2. 带元数据的文档管理")
    print("3. 检索效果对比")
    print()
    
    choice = input("选择演示 (1/2/3/all) [默认: 1]: ").strip() or "1"
    
    if choice == "1":
        demo_basic()
    elif choice == "2":
        demo_with_metadata()
    elif choice == "3":
        demo_comparison()
    elif choice.lower() == "all":
        demo_basic()
        demo_with_metadata()
        demo_comparison()
    else:
        print("无效选择")
        return
    
    print("\n" + "="*70)
    print("🎉 演示完成！")
    print("="*70)
    print("\n💡 核心要点：")
    print("1. RAG = 检索（Retrieval）+ 生成（Generation）")
    print("2. 通过检索相关文档，让LLM生成更准确的答案")
    print("3. 解决了LLM知识过时和幻觉问题")
    print("4. 实际应用需要接入真实的LLM API（如GPT、Claude等）")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()



