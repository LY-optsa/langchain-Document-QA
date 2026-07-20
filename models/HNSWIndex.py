'''
Add HNSWIndex.py
Author: liyu
Date: 2025-09-11
Update:2025-11-10 19:20
UpdateLog:
    In HNSWRAGWrapper, we modified the methord of create HNSW index, which can created multi HNSW index for different documents.
'''

import os
import numpy as np
import faiss
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from models.config import *


class HNSWRetriever:
    """
    基于Faiss HNSW索引的向量检索器
    """
    
    def __init__(self, embeddings_model=None, index_path=None):
        self.embeddings_model = embeddings_model
        self.index = None
        self.documents = []
        self.index_path = index_path
        self.id_to_doc = {}  # ID到文档的映射
        self.next_id = 0    
        
        # 初始化HNSW索引参数
        self.dim = 768  # 默认嵌入维度，可根据实际模型调整
        self.M = 32  # HNSW参数：每个节点的连接数
        self.ef_construction = 200  # HNSW参数：构建时的探索因子
        
        # 如果索引文件存在，则加载索引
        if index_path and os.path.exists(index_path):
            self.load_index()

    def clear_index(self):
        """清空索引"""
        self.index = None
        self.documents = []
        self.id_to_doc = {}
        self.next_id = 0    
    
    def _get_embedding_dimension(self, sample_text="测试文本"):
        """获取嵌入向量的维度"""
        if self.embeddings_model:
            try:
                sample_embedding = self.embeddings_model.embed_query(sample_text)
                self.dim = len(sample_embedding)
            except:
                # 默认使用768维
                self.dim = 768
        return self.dim
    
    def init_index(self):
        """初始化HNSW索引"""
        dim = self._get_embedding_dimension()
        
        # 创建HNSW索引，使用IndexIDMap来支持ID映射
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # 使用IndexIDMap来支持添加ID
        self.index = faiss.IndexIDMap(self.index)
        
        print(f"HNSW索引初始化完成，维度: {dim}, M: {self.M}")
    
    def add_documents(self, documents: List[Document]):
        """添加文档到索引中"""
        if self.index is None:
            self.init_index()
        
        # 生成嵌入向量
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings_model.embed_documents(texts)
        
        # 转换为numpy数组
        embeddings_np = np.array(embeddings).astype('float32')
        
        # 生成唯一的ID
        # start_id = len(self.documents)
        ids = np.arange(self.next_id, self.next_id + len(documents))
        # 更新下一个ID
        self.next_id += len(documents)
        
        # 添加文档到内部存储
        self.documents.extend(documents)
        
        # 添加到索引（使用ID映射）
        self.index.add_with_ids(embeddings_np, ids)
        
        # 建立ID到文档的映射
        for i, doc_id in enumerate(ids):
            self.id_to_doc[doc_id] = documents[i]
        
        print(f"已添加 {len(documents)} 个文档到HNSW索引中")
    
    def search(self, query: str, k: int ) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        if self.index is None or len(self.documents) == 0:
            print("HNSW索引未初始化或文档为空")
            return []
        
        # 生成查询嵌入
        query_embedding = self.embeddings_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # 设置搜索时的探索因子
        # 设置Faiss使用的线程数，提高并发检索性能
        faiss.omp_set_num_threads(8)  # 增加线程数以利用多核CPU
        if hasattr(self.index, 'index') and hasattr(self.index.index, 'hnsw'):
            self.index.index.hnsw.efSearch = 50
        
        # 执行搜索
        scores, indices = self.index.search(query_embedding_np, k)
        
        # 整理结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx in self.id_to_doc:  # 有效索引
                doc = self.id_to_doc[idx]
                # 转换距离为相似度分数（距离越小分数越高）
                similarity_score = 1.0 / (1.0 + float(score))
                results.append((doc, similarity_score))
        
        return results
    
    def save_index(self, path=None):
        """保存索引到磁盘"""
        save_path = path or self.index_path
        if save_path and self.index:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            faiss.write_index(self.index, save_path)
            print(f"HNSW索引已保存到: {save_path}")
    
    def load_index(self, path=None):
        """从磁盘加载索引"""
        load_path = path or self.index_path
        if load_path and os.path.exists(load_path):
            self.index = faiss.read_index(load_path)
            print(f"HNSW索引已从 {load_path} 加载")
            
            # 设置索引参数
            self.dim = self.index.d
            # 重新初始化相关参数
            if hasattr(self.index, 'index') and hasattr(self.index.index, 'hnsw'):
                self.M = self.index.index.hnsw.nb_neighbors(1) if self.index.index.hnsw.nb_neighbors(1) > 0 else 32
            
            # 重新加载文档数据（需要从Chroma获取）
            self._reload_documents()
    
    def _reload_documents(self):
        """重新加载文档数据（从Chroma存储中获取）"""
        if not self.index_path:
            return
            
        # 从索引路径推断Chroma路径
        chroma_path = os.path.dirname(self.index_path)
        if os.path.exists(chroma_path) and chroma_path.endswith('Chroma'):
            try:
                chroma_store = Chroma(persist_directory=chroma_path, embedding_function=self.embeddings_model)
                results = chroma_store.get()
                
                if results and 'documents' in results:
                    # 清空现有文档
                    self.documents = []
                    self.id_to_doc = {}
                    
                    # 重建Document对象列表
                    for i, content in enumerate(results['documents']):
                        doc = Document(page_content=content)
                        if 'metadatas' in results and i < len(results['metadatas']):
                            doc.metadata = results['metadatas'][i]
                        self.documents.append(doc)
                    
                    # 重建ID映射（假设ID是连续的）
                    for i, doc in enumerate(self.documents):
                        self.id_to_doc[i] = doc
                    
                    print(f"已重新加载 {len(self.documents)} 个文档")
            except Exception as e:
                print(f"重新加载文档失败: {e}")


# class HNSWRAGWrapper:
#     """
#     HNSW检索器的RAG包装类，用于与现有系统集成
#     """
    
#     def __init__(self, filepath: str, vs_path: str, embeddings: object, init: bool = True, llm=None):
#         self.filepath = filepath
#         self.vs_path = vs_path
#         self.embeddings = embeddings
#         self.llm = llm
#         self.top_k = VECTOR_SEARCH_TOP_K
        
#         # HNSW索引路径
#         self.hnsw_index_path = os.path.join(vs_path, "hnsw_index.faiss")
        
#         # 初始化HNSW检索器
#         self.hnsw_retriever = HNSWRetriever(embeddings_model=embeddings, index_path=self.hnsw_index_path)
        
#         if init:
#             self._init_knowledge_base()
    
#     def _init_knowledge_base(self):
#         """初始化知识库"""
#         from models.RAGWithChroma import init_knowledge_vector_store
        
#         # 先使用Chroma初始化向量存储
#         vs_path, loaded_files = init_knowledge_vector_store(
#             filepath=self.filepath,
#             vs_path=self.vs_path,
#             embeddings=self.embeddings
#         )
        
#         # 从Chroma加载文档并添加到HNSW索引
#         if vs_path:
#             chroma_store = Chroma(persist_directory=vs_path, embedding_function=self.embeddings)
#             results = chroma_store.get()
            
#             if results and 'documents' in results:
#                 # 构建Document对象列表
#                 documents = []
#                 for i, content in enumerate(results['documents']):
#                     doc = Document(page_content=content)
#                     if 'metadatas' in results and i < len(results['metadatas']):
#                         doc.metadata = results['metadatas'][i]
#                     documents.append(doc)
                
#                 # 添加到HNSW索引
#                 self.hnsw_retriever.add_documents(documents)
                
#                 # 保存索引
#                 self.hnsw_retriever.save_index()
    
#     def query_knowledge(self, query: str) -> str:
#         """查询知识库"""
#         # 使用HNSW检索器搜索
#         results = self.hnsw_retriever.search(query, self.top_k)
        
#         # 提取文档内容
#         contents = [doc.page_content for doc, score in results]
#         related_content = "\n".join(contents)
        
#         return related_content
    
#     def query_knowledge_with_scores(self, query: str) -> List[Tuple[Document, float]]:
#         """查询知识库并返回带分数的结果"""
#         return self.hnsw_retriever.search(query, self.top_k)

class HNSWRAGWrapper:
    """
    HNSW检索器的RAG包装类，用于与现有系统集成
    """
    def __init__(self, filepath: str, vs_path: str, embeddings: object, init: bool = True, llm=None, top_k: int=1, clear_existing: bool = True, concurrent_workers: int = 8):
        self.filepath = filepath
        self.vs_path = vs_path
        self.embeddings = embeddings
        self.llm = llm
        self.top_k = top_k
        self.clear_existing = clear_existing
        self.concurrent_workers = concurrent_workers
        
        # HNSW索引路径
        self.hnsw_index_path = os.path.join(vs_path, "hnsw_index.faiss")
        
        # 初始化HNSW检索器
        self.hnsw_retriever = HNSWRetriever(embeddings_model=embeddings, index_path=self.hnsw_index_path)
        
        # 检查是否已存在向量存储和HNSW索引
        chroma_exists = os.path.exists(vs_path) and os.path.isdir(vs_path)
        hnsw_exists = os.path.exists(self.hnsw_index_path)
        
        # 如果启用了初始化，则删除现有的HNSW索引文件（强制重新创建）
        if init and hnsw_exists:
            print(f"初始化模式下，删除现有的HNSW索引文件: {self.hnsw_index_path}")
            try:
                os.remove(self.hnsw_index_path)
                # 重新初始化HNSW检索器
                self.hnsw_retriever = HNSWRetriever(embeddings_model=embeddings, index_path=self.hnsw_index_path)
                hnsw_exists = False  # 标记为不存在，以便重新创建
                print("HNSW索引文件已删除")
            except Exception as e:
                print(f"删除HNSW索引文件失败: {e}")
        
        # 如果启用了初始化或者向量存储不存在，则初始化知识库
        if init or not chroma_exists:
            self._init_knowledge_base()
        elif chroma_exists and hnsw_exists:
            # 如果向量存储和HNSW索引都存在，则直接加载
            print("检测到已存在的向量存储和HNSW索引，直接加载...")
            # 从Chroma加载文档并添加到HNSW索引
            chroma_store = Chroma(persist_directory=vs_path, embedding_function=self.embeddings)
            results = chroma_store.get()
            
            if results and 'documents' in results:
                # 构建Document对象列表
                documents = []
                for i, content in enumerate(results['documents']):
                    doc = Document(page_content=content)
                    if 'metadatas' in results and i < len(results['metadatas']):
                        doc.metadata = results['metadatas'][i]
                    documents.append(doc)
                
                # 添加到HNSW索引
                self.hnsw_retriever.add_documents(documents)
        elif chroma_exists and not hnsw_exists:
            # 如果只有Chroma向量存储存在但HNSW索引不存在，则从Chroma创建HNSW索引
            print("检测到已存在的向量存储，但HNSW索引不存在，从Chroma创建HNSW索引...")
            self._create_hnsw_from_chroma()
    
    # def __init__(self, filepath: str, vs_path: str, embeddings: object, init: bool = True, llm=None, top_k: int=1):
    #     self.filepath = filepath
    #     self.vs_path = vs_path
    #     self.embeddings = embeddings
    #     self.llm = llm
    #     self.top_k = top_k
        
    #     # HNSW索引路径
    #     self.hnsw_index_path = os.path.join(vs_path, "hnsw_index.faiss")
        
    #     # 初始化HNSW检索器
    #     self.hnsw_retriever = HNSWRetriever(embeddings_model=embeddings, index_path=self.hnsw_index_path)
        
    #     # 检查是否已存在向量存储和HNSW索引
    #     chroma_exists = os.path.exists(vs_path) and os.path.isdir(vs_path)
    #     hnsw_exists = os.path.exists(self.hnsw_index_path)
        
    #     # 如果启用了初始化或者向量存储不存在，则初始化知识库
    #     if init or not chroma_exists:
    #         self._init_knowledge_base()
    #     elif chroma_exists and hnsw_exists:
    #         # 如果向量存储和HNSW索引都存在，则直接加载
    #         print("检测到已存在的向量存储和HNSW索引，直接加载...")
    #         self.hnsw_retriever.clear_index()
    #         # 从Chroma加载文档并添加到HNSW索引
    #         chroma_store = Chroma(persist_directory=vs_path, embedding_function=self.embeddings)
    #         results = chroma_store.get()
            
    #         if results and 'documents' in results:
    #             # 构建Document对象列表
    #             documents = []
    #             for i, content in enumerate(results['documents']):
    #                 doc = Document(page_content=content)
    #                 if 'metadatas' in results and i < len(results['metadatas']):
    #                     doc.metadata = results['metadatas'][i]
    #                 documents.append(doc)
                
    #             # 添加到HNSW索引
    #             self.hnsw_retriever.add_documents(documents)
    #     elif chroma_exists and not hnsw_exists:
    #         # 如果只有Chroma向量存储存在但HNSW索引不存在，则从Chroma创建HNSW索引
    #         print("检测到已存在的向量存储，但HNSW索引不存在，从Chroma创建HNSW索引...")
    #         self.hnsw_retriever.clear_index()   
    #         self._create_hnsw_from_chroma()
    
    # def _init_knowledge_base(self):
    #     """初始化知识库"""
    #     from models.HNSWRAG import init_knowledge_vector_store

    #     if os.path.exists(self.hnsw_index_path):
    #         print(f"删除现有的HNSW索引文件: {self.hnsw_index_path}")
    #         try:
    #             os.remove(self.hnsw_index_path)
    #             # 重新初始化HNSW检索器
    #             self.hnsw_retriever = HNSWRetriever(embeddings_model=self.embeddings, index_path=self.hnsw_index_path)
    #             print("HNSW索引文件已删除")
    #         except Exception as e:
    #             print(f"删除HNSW索引文件失败: {e}")
        
    #     # 先使用Chroma初始化向量存储
    #     vs_path, loaded_files = init_knowledge_vector_store(
    #         filepath=self.filepath,
    #         vs_path=self.vs_path,
    #         embeddings=self.embeddings
    #     )

    #     self.hnsw_retriever.clear_index()
        
    #     # 从Chroma加载文档并添加到HNSW索引
    #     if vs_path:
    #         chroma_store = Chroma(persist_directory=vs_path, embedding_function=self.embeddings)
    #         results = chroma_store.get()
            
    #         if results and 'documents' in results:
    #             # 构建Document对象列表
    #             documents = []
    #             for i, content in enumerate(results['documents']):
    #                 doc = Document(page_content=content)
    #                 if 'metadatas' in results and i < len(results['metadatas']):
    #                     doc.metadata = results['metadatas'][i]
    #                 documents.append(doc)
                
    #             # 添加到HNSW索引
    #             self.hnsw_retriever.add_documents(documents)
                
    #             # 保存索引
    #             self.hnsw_retriever.save_index()

    def _init_knowledge_base(self):
        """初始化知识库"""
        from models.HNSWRAG import init_knowledge_vector_store

        # 删除现有的HNSW索引文件
        if os.path.exists(self.hnsw_index_path):
            print(f"删除现有的HNSW索引文件: {self.hnsw_index_path}")
            try:
                os.remove(self.hnsw_index_path)
                print("HNSW索引文件已删除")
            except Exception as e:
                print(f"删除HNSW索引文件失败: {e}")
        
        # 重新初始化HNSW检索器，确保next_id重置
        self.hnsw_retriever = HNSWRetriever(embeddings_model=self.embeddings, index_path=self.hnsw_index_path)
        
        # 先使用Chroma初始化向量存储，根据参数决定是否清空现有存储
        vs_path, loaded_files = init_knowledge_vector_store(
            filepath=self.filepath,
            vs_path=self.vs_path,
            embeddings=self.embeddings,
            clear_existing=self.clear_existing,
            concurrent_workers=self.concurrent_workers
        )

        # 从Chroma加载文档并添加到HNSW索引
        if vs_path:
            chroma_store = Chroma(persist_directory=vs_path, embedding_function=self.embeddings)
            results = chroma_store.get()
            
            # 显式关闭Chroma实例，释放文件句柄
            if hasattr(chroma_store, '_client'):
                chroma_store._client = None
            if hasattr(chroma_store, '_persist_directory'):
                chroma_store._persist_directory = None
            chroma_store = None
            
            if results and 'documents' in results:
                # 构建Document对象列表
                documents = []
                for i, content in enumerate(results['documents']):
                    doc = Document(page_content=content)
                    if 'metadatas' in results and i < len(results['metadatas']):
                        doc.metadata = results['metadatas'][i]
                    documents.append(doc)
                
                # 添加到HNSW索引
                self.hnsw_retriever.add_documents(documents)
                
                # 保存索引
                self.hnsw_retriever.save_index()
    
    # def _create_hnsw_from_chroma(self):
    #     """从已有的Chroma向量存储创建HNSW索引"""
    #     if os.path.exists(self.hnsw_index_path):
    #         print(f"删除现有的HNSW索引文件: {self.hnsw_index_path}")
    #         try:
    #             os.remove(self.hnsw_index_path)
    #             # 重新初始化HNSW检索器
    #             self.hnsw_retriever = HNSWRetriever(embeddings_model=self.embeddings, index_path=self.hnsw_index_path)
    #             print("HNSW索引文件已删除")
    #         except Exception as e:
    #             print(f"删除HNSW索引文件失败: {e}")

    #     self.hnsw_retriever.clear_index()
    #     if os.path.exists(self.vs_path):
    #         chroma_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
    #         results = chroma_store.get()
            
    #         if results and 'documents' in results:
    #             # 构建Document对象列表
    #             documents = []
    #             for i, content in enumerate(results['documents']):
    #                 doc = Document(page_content=content)
    #                 if 'metadatas' in results and i < len(results['metadatas']):
    #                     doc.metadata = results['metadatas'][i]
    #                 documents.append(doc)
                
    #             # 添加到HNSW索引
    #             self.hnsw_retriever.add_documents(documents)
                
    #             # 保存索引
    #             self.hnsw_retriever.save_index()

    def _create_hnsw_from_chroma(self):
        """从已有的Chroma向量存储创建HNSW索引"""
        # 删除现有的HNSW索引文件
        if os.path.exists(self.hnsw_index_path):
            print(f"删除现有的HNSW索引文件: {self.hnsw_index_path}")
            try:
                os.remove(self.hnsw_index_path)
                print("HNSW索引文件已删除")
            except Exception as e:
                print(f"删除HNSW索引文件失败: {e}")

        # 重新初始化HNSW检索器，确保next_id重置
        self.hnsw_retriever = HNSWRetriever(embeddings_model=self.embeddings, index_path=self.hnsw_index_path)
        
        if os.path.exists(self.vs_path):
            chroma_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
            results = chroma_store.get()
            
            # 显式关闭Chroma实例，释放文件句柄
            if hasattr(chroma_store, '_client'):
                chroma_store._client = None
            if hasattr(chroma_store, '_persist_directory'):
                chroma_store._persist_directory = None
            chroma_store = None
            
            if results and 'documents' in results:
                # 构建Document对象列表
                documents = []
                for i, content in enumerate(results['documents']):
                    doc = Document(page_content=content)
                    if 'metadatas' in results and i < len(results['metadatas']):
                        doc.metadata = results['metadatas'][i]
                    documents.append(doc)
                
                # 添加到HNSW索引
                self.hnsw_retriever.add_documents(documents)
                
                # 保存索引
                self.hnsw_retriever.save_index()
    
    def query_knowledge(self, query: str) -> str:
        """查询知识库"""
        # 使用HNSW检索器搜索
        results = self.hnsw_retriever.search(query, self.top_k)
        
        # 提取文档内容
        contents = [doc.page_content for doc, score in results]
        related_content = "\n".join(contents)
        
        return related_content
    
    def query_knowledge_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """查询知识库并返回带分数的结果"""
        return self.hnsw_retriever.search(query, self.top_k)