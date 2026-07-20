#coding=utf-8
'''
BM25 retrieval model with multi-granularity tokenization using jieba.
Author: liyu
Date: 2025-09-11
update:2025-11-06 18:35 
'''

import math
from collections import defaultdict, Counter
import jieba
import re

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75, delta=0.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        # 并行预处理文档，提高初始化速度
        self.corpus = []
        self.doc_lengths = []
        self.doc_term_counts = []
        
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.corpus.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.doc_term_counts.append(Counter(tokens))
        
        self.avg_dl = sum(self.doc_lengths) / len(self.corpus) if self.corpus else 0
        self.inverted_index = self._build_inverted_index()
        self.idf = self._compute_idf()

    def _preprocess_text(self, text):
        """预处理文本"""
        return re.sub(r'[a-zA-Z]+', lambda m: m.group().lower(), text)

    def _tokenize(self, text):
        """多粒度分词，优化性能"""
        # 预处理文本
        preprocessed_text = self._preprocess_text(text)
        
        # 使用jieba快速模式进行分词
        terms = list(jieba.cut(preprocessed_text, cut_all=False))
        
        # 生成二元语法（bigrams）
        ngrams = []
        for i in range(len(terms)-1):
            ngrams.append(terms[i] + terms[i+1])
        
        # 使用集合去重，但保留顺序以提高缓存命中率
        result = []
        seen = set()
        for item in terms + ngrams:
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        return result

    def _build_inverted_index(self):
        """优化的倒排索引构建，避免重复文档ID"""
        inverted_index = defaultdict(set)  # 使用set避免重复
        
        for doc_id, terms in enumerate(self.corpus):
            for term in terms:
                inverted_index[term].add(doc_id)
        
        # 将set转换为list以保持兼容性
        return {term: list(doc_ids) for term, doc_ids in inverted_index.items()}

    def _compute_idf(self):
        """优化的IDF计算，使用平滑处理"""
        idf = {}
        N = len(self.corpus)
        log_N_plus_1 = math.log(N + 1)  # 预计算以提高效率
        
        for term, docs in self.inverted_index.items():
            df = len(docs)
            # 优化的IDF计算，使用拉普拉斯平滑
            idf[term] = log_N_plus_1 - math.log(df + 1)
        
        return idf

    def get_scores(self, query):
        """优化的评分计算，减少重复计算"""
        query_terms = self._multi_granularity_tokenize(query)
        scores = [0.0] * len(self.corpus)
        
        # 预计算文档长度因子以减少重复计算
        doc_length_factors = []
        for doc_len in self.doc_lengths:
            # 优化的文档长度因子计算
            if self.avg_dl > 0:
                doc_length_factors.append(self.delta + 1 - self.b + self.b * (doc_len / self.avg_dl))
            else:
                doc_length_factors.append(self.delta + 1)
        
        # 只处理查询中的词项
        for term in set(query_terms):  # 去重以避免重复计算
            idf_val = self.idf.get(term, 0)  # 未登录词idf设为0
            if idf_val == 0:
                continue
            
            # 只计算包含该词项的文档
            for doc_id in range(len(self.corpus)):
                doc_term_counts = self.doc_term_counts[doc_id]
                tf = doc_term_counts.get(term, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * doc_length_factors[doc_id]
                    scores[doc_id] += idf_val * (numerator / denominator)
        
        return scores

    def _multi_granularity_tokenize(self, query):
        """复用_tokenize方法，保持一致性"""
        return self._tokenize(query)

    def search(self, query, top_n=5):
        """执行搜索并返回top_n个结果，优化排序性能"""
        scores = self.get_scores(query)
        
        # 使用列表推导式和sorted优化排序
        scored_docs = [(doc_id, score) for doc_id, score in enumerate(scores) if score > 0]
        
        # 仅对有正分数的文档进行排序
        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        return ranked_docs[:top_n]

# 示例用法
if __name__ == "__main__":
    # 示例文档集（每个文档是分词后的词列表）
    corpus = [
    "今天天气真好，适合出去游玩。",
    "机器学习是人工智能的重要分支。",
    "Python编程语言广泛应用于数据分析，这是一种简单易用的编程语言。",
    "C++是一种高性能的编程语言。",
    "深度学习在图像识别中表现出色。"
]
    # 初始化BM25模型
    bm25 = BM25(corpus)
    print(bm25.corpus)
    
    # 执行搜索
    query = "编程语言"
    results = bm25.search(query, top_n=2)
    print(results)
    
    # 打印结果
    print(f"搜索查询: '{query}'")
    print("Top 3 结果:")
    for rank, (doc_id, score) in enumerate(results, 1):
        original_text = "".join(corpus[doc_id])  # 重建原始文本
        print(f"{rank}. [文档{doc_id}] 得分: {score:.4f}")
        print(f"   内容: {original_text}")
