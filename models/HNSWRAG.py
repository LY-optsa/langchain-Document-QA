#coding=utf-8
'''
Update:2025-12-10 20:00
author: Liyu
logs:
  1. Update vector store init function to support update vector store
  2. Modified RAG class
'''

from typing import List
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.indexes import SQLRecordManager, index
from models.TextSplitter import ChineseTextSplitter, WindowedTextSplitter, BasicTextSplitter, EnhancedDocumentSplitter
from models.BM25_retrieval import BM25
from models.LarkSuiteDocProcess import *
from langchain_chroma import Chroma
import datetime
import torch
from tqdm import tqdm
import pandas as pd
from models.config import *
from sentence_transformers import CrossEncoder
from typing import *
import shutil
import time
import gc
import hashlib
#import logging

# 配置日志记录器
#logger = logging.getLogger(__name__)

# 导入HNSW索引支持
try:
    from models.HNSWIndex import HNSWRAGWrapper
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    #logger.warning("HNSW索引模块未找到，将使用默认Chroma检索")

def load_larksuite_doc(urls: list[str], custom_metadata: dict = None) -> list[Document]:
    """
    加载飞书文档并支持自定义元数据
    
    Args:
        urls: 飞书文档URL列表
        custom_metadata: 自定义元数据字典，将添加到每个文档的元数据中
    
    Returns:
        Document对象列表
    """
    documents = []
    for url in urls:
        name, token = get_table_name(url)
        sheets = get_sheet(token)
        for sheet in sheets['data']['sheets']:
            sheet_id = sheet['sheet_id']
            text_list = preprocess_karksuite_doc(token, sheet_id)
            title = sheet['title']
            
            # 将文本列表转换为Document对象
            for text in text_list:
                # 创建基础元数据
                metadata = {
                    'title': title,
                    'document_name': name
                }
                
                # 添加自定义元数据（如果提供）
                if custom_metadata:
                    metadata.update(custom_metadata)
                
                # 创建Document对象
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
    
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        keep_separator=True  # 保留原始分隔符
    )
            
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def load_txt_file(filepath: Any) -> (list[Document] | list):
    try:
        from charset_normalizer import detect
        with open(filepath, 'rb') as f:
            rawdata = f.read()
            result = detect(rawdata)
            
        # 使用更健壮的文件加载方式
        loader = TextLoader(filepath, 
                          encoding=result['encoding'] if result['encoding'] else 'utf-8-sig',
                          autodetect_encoding=True)
        
        # 添加容错的分块策略
        textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            keep_separator=True  # 保留原始分隔符
        )
        docs = loader.load_and_split(text_splitter=textsplitter)
        
        # 为每个文本块添加文件名元数据
        filename = os.path.basename(filepath)
        for doc in docs:
            doc.metadata['filename'] = filename
            
        return docs
    except Exception as e:
        #logger.error(f"文件加载失败: {str(e)}")
        return []
    
def load_excel_file(filepath: str) -> list[Document]:
    try:
        excel = pd.ExcelFile(filepath)
        all_sheets = excel.sheet_names
        sheet_data = {sheet_name: pd.read_excel(excel, sheet_name) for sheet_name in all_sheets}
        texts = {}
        for sheet_name, df in sheet_data.items():
            texts[sheet_name] = df.to_string()
        text_splitter = EnhancedDocumentSplitter(
            chunk_size=5000,
            chunk_overlap=100,
            document_type="excel",
        )
        texts_chunks = {}
        for sheet_name, text in texts.items():
            texts_chunks[sheet_name] = text_splitter.split_text(text)
        documents = []
        for sheet_name, chunks in texts_chunks.items():
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": sheet_name, "filename": os.path.basename(filepath)}))
        return documents
    except Exception as e:
        #logger.error(f"文件加载失败: {str(e)}")
        return []


def torch_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            information = f"{e} 如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。"
            # logger.error(f"{e}")
            # logger.info("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


def load_file(filepath: str, is_larksuite=False) -> list[Document]:
    # 获取文件名
    filename = os.path.basename(filepath)
    
    if filepath.lower().endswith(".md"):
        loader = UnstructuredMarkdownLoader(filepath, mode="elements")
        textsplitter = WindowedTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = loader.load()
        docs = filter_complex_metadata(docs)
        docs = textsplitter.split_documents(docs)
        for doc in docs:
            doc.metadata['filename'] = filename
    elif filepath.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        textsplitter = EnhancedDocumentSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, document_type="pdf")
        docs = loader.load_and_split(textsplitter)
        docs = filter_complex_metadata(docs)
        for doc in docs:
            doc.metadata['filename'] = filename

    elif filepath.lower().endswith(".xlsx") or filepath.lower().endswith(".xls"):
        docs = load_excel_file(filepath)

    elif is_larksuite:
        url = [
            "https://cgu3k3reqx.feishu.cn/sheets/JtIUs8sIUhIVe3tYSJMcUZiLncm",
            "https://cgu3k3reqx.feishu.cn/sheets/IYuksoXvJhjoAatRh0DcE5BInPf",
            "https://cgu3k3reqx.feishu.cn/sheets/XL39s5VVUhdRPxtfK8BcHg8pnje"
        ]
        docs = load_larksuite_doc(url)
    else:
        docs = load_txt_file(filepath)
        docs = filter_complex_metadata(docs)
        for doc in docs:
            doc.metadata['filename'] = filename
    
    # 为每个文本块添加文件名元数据
    # for doc in docs:
    #     doc.metadata['filename'] = filename
    
    return docs


def get_related_content(related_docs: Any) -> LiteralString:
    related_content = []
    for doc in related_docs:
        # 优先检查 LarkSuite 文档的元数据字段
        filename = doc.metadata.get("document_name") or doc.metadata.get("filename", "未知文件")
        sheet_title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")
        
        # 如果有表格标题，则显示文件名和表格名
        if sheet_title:
            content_with_filename = f"文件: {filename}\nSheet: {sheet_title}\n内容: {doc.page_content}"
        # 如果有来源信息且不是 URL 或绝对路径，则显示文件名和来源
        elif source and not source.startswith("http://") and not source.startswith("https://") and not os.path.isabs(source):
            content_with_filename = f"文件: {filename}\nSheet: {source}\n内容: {doc.page_content}"
        else:
            content_with_filename = f"文件: {filename}\n内容: {doc.page_content}"
            
        related_content.append(content_with_filename)
    return "\n".join(related_content)

def get_docs_with_score(docs_with_score: Any) -> List:
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs

# def init_knowledge_vector_store(filepath: str | List[str],
#                                 vs_path: str | os.PathLike = None,
#                                 embeddings: object = None,
#                                 concurrent_workers: int = 4) -> (tuple[str | os.PathLike, list] | tuple[None, list] | None):
#     """
#     初始化知识库向量存储，支持并发文件加载
    
#     Args:
#         filepath: 文件路径或文件路径列表
#         vs_path: 向量存储路径
#         embeddings: 嵌入模型
#         concurrent_workers: 并发工作线程数
    
#     Returns:
#         向量存储路径和加载的文件列表
#     """
#     import concurrent.futures
#     from concurrent.futures import ThreadPoolExecutor
#     import threading
    
#     # 添加线程锁以保护共享资源
#     lock = threading.Lock()
#     loaded_files = []
#     failed_files = []
#     all_docs = []
    
#     def process_single_file(file_path):
#         """处理单个文件的函数，用于并发执行"""
#         nonlocal all_docs
#         try:
#             file_docs = load_file(file_path)
#             with lock:
#                 all_docs.extend(file_docs)
#                 loaded_files.append(file_path)
#                 print(f"{os.path.split(file_path)[-1]} 已成功加载")
#             return True
#         except Exception as e:
#             with lock:
#                 print(e)
#                 print(f"{os.path.split(file_path)[-1]} 未能成功加载")
#                 failed_files.append(os.path.split(file_path)[-1])
#             return False
    
#     # 准备要处理的文件列表
#     files_to_process = []
    
#     # 单个文件
#     if isinstance(filepath, str):
#         if not os.path.exists(filepath):
#             print(f"{filepath} 路径不存在")
#             return None
#         elif os.path.isfile(filepath):
#             files_to_process.append(filepath)
#         elif os.path.isdir(filepath):
#             files_to_process = [os.path.join(filepath, f) for f in os.listdir(filepath)]
#     # 文件列表
#     else:
#         files_to_process = filepath
    
#     # 使用线程池并发处理文件
#     if files_to_process:
#         print(f"开始并发加载 {len(files_to_process)} 个文件，使用 {concurrent_workers} 个工作线程...")
#         with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
#             # 使用tqdm显示进度
#             with tqdm(total=len(files_to_process), desc="加载文件") as pbar:
#                 futures = {executor.submit(process_single_file, f): f for f in files_to_process}
#                 for future in concurrent.futures.as_completed(futures):
#                     pbar.update(1)
    
#     if len(failed_files) > 0:
#         print("以下文件未能成功加载：")
#         for file in failed_files:
#             print(file)

#     if len(all_docs) > 0:
#         logger.info("文件加载完毕，正在生成向量库")
#         if vs_path and os.path.isdir(vs_path):
#             # vector_store = FAISS.load_local(vs_path, embeddings)
#             vector_store = Chroma(persist_directory=vs_path, embedding_function=embeddings)
#             vector_store.add_documents(all_docs)
#             torch_gc()
#         else:
#             if not vs_path:
#                 vs_path = os.path.join("\tmp",
#                                        f"""CHROMA{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
#             vector_store = Chroma.from_documents(all_docs, embeddings, persist_directory=vs_path)
#             torch_gc()

#         # vector_store.save_local(vs_path)
#         print("向量生成成功")
#         return vs_path, loaded_files
#     else:
#         print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
#         return None, loaded_files

def safe_remove_directory(path, max_retries=3, delay=2):
    """安全删除目录，处理Windows文件锁定问题"""
    if not os.path.exists(path):
        #logger.info(f"目录不存在: {path}")
        return True
    
    # 先尝试使用普通方法删除
    try:
        shutil.rmtree(path)
        #logger.info(f"成功删除目录: {path}")
        return True
    except PermissionError:
        information = f"删除目录失败，文件可能被锁定: {path}"
        #logger.warning(f"删除目录失败，文件可能被锁定: {path}")
    except Exception as e:
        #logger.error(f"删除目录时发生意外错误: {e}")
        return False
    
    # 如果普通删除失败，尝试强制垃圾回收和重试
    for i in range(max_retries):
        try:
            # 强制垃圾回收，释放可能的资源
            gc.collect()
            torch_gc()
            
            # 短暂延迟
            time.sleep(delay)
            
            # 尝试删除目录
            shutil.rmtree(path)
            #logger.info(f"重试 {i+1}/{max_retries} 成功删除目录: {path}")
            return True
        except PermissionError as e:
            #logger.warning(f"重试 {i+1}/{max_retries} 删除目录失败: {e}")
            # 增加延迟时间
            delay *= 1.5
        except Exception as e:
            #logger.error(f"重试 {i+1}/{max_retries} 发生意外错误: {e}")
            return False
    
    # 如果所有重试都失败，尝试重命名目录（作为备选方案）
    try:
        import uuid
        new_path = path + "_deleted_" + str(uuid.uuid4())
        os.rename(path, new_path)
       # logger.warning(f"无法直接删除目录，已将其重命名为: {new_path}")
        return True
    except Exception as e:
       # logger.error(f"重命名目录也失败: {e}")
        return False
    
def init_larksuite_vector_store(url: List[str], vs_path: str | os.PathLike = None,
                                embeddings: object = None,clear_existing: bool = False) -> (tuple[str | os.PathLike, list] | tuple[None, list] | None):
    loader_files = load_larksuite_doc(url)
    if len(loader_files) > 0:

       # logger.info("文件加载完毕，正在生成向量库")
        if vs_path and os.path.isdir(vs_path):
            vector_store = Chroma(persist_directory=vs_path, embedding_function=embeddings)
            vector_store.add_documents(loader_files)
            torch_gc()

        else:
            if not vs_path:
                vs_path = os.path.join("\tmp",
                                       f"""CHROMA{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
            vector_store = Chroma.from_documents(loader_files, embeddings, persist_directory=vs_path)
            torch_gc()
        return vs_path, loader_files
    
    else:
        return None, loader_files
    
def update_larksuite_vector_store(embedding, url: List[str], vs_path: str | os.PathLike = None, delete_existing: bool = False):
    loader_files = load_larksuite_doc(url)
    if len(loader_files) > 0: 
        vector_store = Chroma(persist_directory=vs_path, embedding_function=embedding, collection_name="larksuite_docs")
        namespace_hash = hashlib.md5(vs_path.encode()).hexdigest()[:16]
        namespace = f"chroma/{namespace_hash}"
        # namespace = "chroma/my_incremental_collection"
        record_manager = SQLRecordManager(
            namespace,
            db_url="sqlite:///record_manager.sqlite"  # 记录管理信息将存储在此SQLite文件
        )
        record_manager.create_schema()
        clean_up_mode = "full" if delete_existing else "incremental"
        result = index(loader_files, record_manager=record_manager, vector_store=vector_store, cleanup=clean_up_mode, source_id_key="document_name")
        return result
    
    else:
        return None
                
            

def init_knowledge_vector_store(filepath: str | List[str],
                                vs_path: str | os.PathLike = None,
                                embeddings: object = None,
                                concurrent_workers: int = 4,
                                clear_existing: bool = False) -> (tuple[str | os.PathLike, list] | tuple[None, list] | None):
    """
    初始化知识库向量存储，支持并发文件加载
    
    Args:
        filepath: 文件路径或文件路径列表
        vs_path: 向量存储路径
        embeddings: 嵌入模型
        concurrent_workers: 并发工作线程数
        clear_existing: 是否清空现有的向量存储
    
    Returns:
        向量存储路径和加载的文件列表
    """
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import shutil
    import time
    import gc
    
    # 添加线程锁以保护共享资源
    lock = threading.Lock()
    loaded_files = []
    failed_files = []
    all_docs = []
    
    def process_single_file(file_path):
        """处理单个文件的函数，用于并发执行"""
        nonlocal all_docs
        try:
            file_docs = load_file(file_path)
            with lock:
                all_docs.extend(file_docs)
                loaded_files.append(file_path)
                #logger.info(f"{os.path.split(file_path)[-1]} 已成功加载")
            return True
        except Exception as e:
            with lock:
                #logger.error(f"文件加载失败: {str(e)}")
                #logger.warning(f"{os.path.split(file_path)[-1]} 未能成功加载")
                failed_files.append(os.path.split(file_path)[-1])
            return False
    
    # 准备要处理的文件列表
    files_to_process = []
    
    # 单个文件
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            #logger.error(f"{filepath} 路径不存在")
            return None
        elif os.path.isfile(filepath):
            files_to_process.append(filepath)
        elif os.path.isdir(filepath):
            files_to_process = [os.path.join(filepath, f) for f in os.listdir(filepath)]
    # 文件列表
    else:
        files_to_process = filepath
    
    # 使用线程池并发处理文件
    if files_to_process:
        #logger.info(f"开始并发加载 {len(files_to_process)} 个文件，使用 {concurrent_workers} 个工作线程...")
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            # 使用tqdm显示进度
            with tqdm(total=len(files_to_process), desc="加载文件") as pbar:
                futures = {executor.submit(process_single_file, f): f for f in files_to_process}
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
    
    if len(failed_files) > 0:
        #logger.warning("以下文件未能成功加载：")
        for file in failed_files:
            information = f"{file} 未能成功加载"
            #logger.warning(f"  {file}")

    if len(all_docs) > 0:
        #logger.info("文件加载完毕，正在生成向量库")
        if vs_path and os.path.isdir(vs_path):
            vector_store = Chroma(persist_directory=vs_path, embedding_function=embeddings)
            vector_store.add_documents(all_docs)
            torch_gc()
            
        else:
            if not vs_path:
                vs_path = os.path.join("\tmp",
                                       f"""CHROMA{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
            vector_store = Chroma.from_documents(all_docs, embeddings, persist_directory=vs_path)
            torch_gc()
        return vs_path, loaded_files
    
    else:
        return None, loaded_files
    
def update_knowledge_vector_store(embedding, filepath: str | os.PathLike, vs_path: str | os.PathLike, delete_existing: bool = False):
    loaded_files = []
    failed_files = []
    # 单个文件
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            return None
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            try:
                docs = load_file(filepath)
                loaded_files.append(filepath)
            except Exception as e:
                return None
        elif os.path.isdir(filepath):
            docs = []
            for file in tqdm(os.listdir(filepath), desc="load files"):
                fullfilepath = os.path.join(filepath, file)

                try:
                    docs += load_file(fullfilepath)
                    loaded_files.append(fullfilepath)
                except Exception as e:
                    failed_files.append(file)

            if len(failed_files) > 0:
                for file in failed_files:
                    information = f"{file} 未能成功加载"
    #  文件列表
    else:
        docs = []
        for file in filepath:
            try:
                docs += load_file(file)
                print(f"{file} 已成功加载")
                loaded_files.append(file)
            except Exception as e:
                print(e)
                print(f"{file} 未能成功加载")

    if len(docs) > 0:
        vector_store = Chroma.from_documents(docs, embedding=embedding, persist_directory=vs_path)
        namespace = "chroma/my_incremental_collection"
        record_manager = SQLRecordManager(
            namespace,
            db_url="sqlite:///record_manager.sqlite"  # 记录管理信息将存储在此SQLite文件
        )
        record_manager.create_schema()
        if delete_existing:
            result = index(docs, record_manager=record_manager, vector_store=vector_store, cleanup="full", source_id_key="filename")
        result = index(docs, record_manager=record_manager, vector_store=vector_store, cleanup="incremental", source_id_key="filename")
        return result
    
    else:
        return None


class BasicRAG:
    def __init__(self):
        pass
    def query_knowledge(self, query: str) -> str:
        pass


class RAG(BasicRAG):
    filepath: str
    vs_path: str
    load_files: List[str] = []
    top_k: int
    embedding: object
    llm: object
    conversation_with_summary: object
    init: bool = True
    vector_store: object = None
    bm25: object = None
    all_documents: List = None
    hnsw_wrapper: object = None  # HNSW包装器

    def __init__(self, filepath: str, vs_path: str, embeddings: object,
                       init: bool = True, llm = None, use_hnsw: bool = False,
                       concurrent_workers: int = 8, top_k: int = 1, clear_existing: bool = True) -> None:
        self.use_hnsw = use_hnsw
        self.concurrent_workers = concurrent_workers
        self.previous_query = None  # 存储用户的上一次提问
        self.last_query_type = None  # 存储上一次查询的类型
        
        # 初始化向量存储为None
        self.vector_store = None
        
        if init:
            # 保存原始vs_path，用于后续清理
            original_vs_path = vs_path
            
            # 初始化向量库，这会返回更新后的vs_path
            vs_path, loaded_files = init_knowledge_vector_store(filepath=filepath,
                                                               embeddings=embeddings,
                                                               vs_path=vs_path,
                                                               concurrent_workers=concurrent_workers,
                                                               clear_existing=clear_existing)
        else:
            # 使用传入的vs_path参数，而不是硬编码的全局变量
            loaded_files = []

        self.load_files = loaded_files
        self.vs_path = vs_path
        self.filepath = filepath
        self.embeddings = embeddings
        self.top_k = top_k
        self.llm = llm
        
        # 初始化HNSW索引（如果启用）
        if self.use_hnsw and HNSW_AVAILABLE:
            try:
                self.hnsw_wrapper = HNSWRAGWrapper(
                    filepath=filepath,
                    vs_path=vs_path,
                    embeddings=embeddings,
                    init=init,
                    llm=llm,
                    top_k=top_k,
                    clear_existing=clear_existing,
                    concurrent_workers=self.concurrent_workers
                )
               # logger.info("HNSW索引已启用")
            except Exception as e:
               # logger.error(f"HNSW索引初始化失败: {e}")
                self.hnsw_wrapper = None
                self.use_hnsw = False
        else:
            self.hnsw_wrapper = None
        
        # 初始化向量存储实例一次，避免每次查询都重新加载
        self.vector_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
        
        # 初始化重排序器
        # self.reranker = CrossEncoder(model_name_or_path='./ReRankers', device='cpu')
        
        # 预加载所有文档并初始化BM25
        self._preload_documents()
    
    def hybrid_retrieval(self, similarity_results, mmr_results, top_k) -> list[tuple]:
        score_map = {}
        for doc, score in similarity_results:
            content_hash = hash(doc.page_content)
            score_map[content_hash] = (doc, score * 0.7)  # 
    
        # 处理MMR结果（假设mmr_results是Document列表）
        for doc in mmr_results:
            content_hash = hash(doc.page_content)
            mmr_score = doc.metadata.get('relevance', 0) * 0.3
            if content_hash in score_map:
                stored_doc, stored_score = score_map[content_hash]
                score_map[content_hash] = (stored_doc, stored_score + mmr_score)
            else:
                score_map[content_hash] = (doc, mmr_score)
    
        # 按分数排序并返回文档对象
        sorted_docs = sorted(score_map.values(), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in sorted_docs[:top_k]]

    
    def query_knowledge(self, query: str) -> LiteralString:
        # 如果启用了HNSW索引且可用，则使用HNSW检索
        if self.use_hnsw and self.hnsw_wrapper:
            try:
                return self.hnsw_wrapper.query_knowledge(query)
            except Exception as e:
                information = f"HNSW检索失败，回退到Chroma检索: {e}"
                #logger.error(f"HNSW检索失败，回退到Chroma检索: {e}")
        
        # vector_store = FAISS.load_local(self.vs_path, self.embeddings)
        vector_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
        actual_top_k = min(self.top_k, vector_store._collection.count())
        # 添加元数据过滤条件
        # similarity_docs = vector_store.similarity_search_with_score(query, k=self.top_k)
        similarity_docs = vector_store.similarity_search_with_score(query, k=actual_top_k)
        mmr_docs = vector_store.max_marginal_relevance_search(query, k=actual_top_k)
        combined_docs = self.hybrid_retrieval(similarity_docs, mmr_docs, actual_top_k)

        # query_doc_pairs = [(query, doc.page_content) for doc, _ in combined_docs]
        # scores = self.reranker.predict(query_doc_pairs)
        # reranked = sorted(zip([doc for doc, _ in combined_docs], scores), 
        #                  key=lambda x: x[1], reverse=True)[:1]  #
        

        # related_docs = get_docs_with_score(combined_docs)
        # vector_store.chunk_size = CHUNK_SIZE
        # related_docs_with_score = vector_store.similarity_search_with_score(query, k = self.top_k)
        # related_docs = get_docs_with_score(related_docs_with_score)
        # related_content = get_related_content(related_docs)
        # return related_content
        return get_related_content([doc for doc, _ in combined_docs])
    
    def query_knowledge_hnsw_with_scores(self, query: str):
        """使用HNSW索引查询并返回带分数的结果"""
        if self.use_hnsw and self.hnsw_wrapper:
            try:
                return self.hnsw_wrapper.query_knowledge_with_scores(query)
            except Exception as e:
                information = f"HNSW检索失败: {e}"
                #logger.error(f"HNSW检索失败: {e}")
            return []
        else:
            #logger.warning("HNSW索引未启用或不可用")
            return []
    
    def _preload_documents(self):
        # 预加载所有文档内容用于BM25初始化
        self.all_documents = []
        if self.vector_store:
            # 获取所有文档内容
            results = self.vector_store.get()
            if results and 'documents' in results:
                # 构建完整的Document对象
                for i, content in enumerate(results['documents']):
                    doc = Document(page_content=content)
                    if 'metadatas' in results and i < len(results['metadatas']):
                        doc.metadata = results['metadatas'][i]
                    self.all_documents.append(doc)
                
                # 初始化BM25模型
                corpus = [doc.page_content for doc in self.all_documents]
                self.bm25 = BM25(corpus)
    
    def query_knowledge_BM25(self, query: str):
        # 检查当前查询是否为"自己已有知识回答以上问题"
        if query == "自己已有知识回答以上问题":
            # 对于此类查询，不召回上下文，返回空字符串
            self.last_query_type = "knowledge_only"
            return ""
        
        # 存储当前查询作为上一次提问
        self.previous_query = query
        self.last_query_type = "normal"
        
        # 如果启用了HNSW索引且可用，则使用HNSW检索
        if self.use_hnsw and self.hnsw_wrapper:
            try:
                return self.hnsw_wrapper.query_knowledge(query)
            except Exception as e:
                information = f"HNSW检索失败，回退到Chroma检索: {e}"
                #logger.error(f"HNSW检索失败，回退到Chroma检索: {e}")
        
        # 使用预初始化的向量存储实例
        actual_top_k = min(self.top_k, self.vector_store._collection.count())
        related_docs_with_score = self.vector_store.similarity_search_with_score(query, k=actual_top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        
        # 使用预初始化的BM25模型
        if self.bm25 and self.all_documents:
            # 执行BM25检索
            bm25_results = self.bm25.search(query, top_n=len(self.all_documents))
            
            # 混合检索结果 (向量权重0.6, BM25权重0.4)
            # 调整权重以提高检索质量
            hybrid_scores = {}
            
            # 为向量检索结果创建快速查找映射
            vector_docs_map = {hash(doc.page_content): doc for doc in related_docs}
            
            # 处理向量检索结果
            for doc in related_docs:
                content_hash = hash(doc.page_content)
                vector_score = doc.metadata.get("score", 0)
                hybrid_scores[content_hash] = (doc, vector_score * 0.6)
            
            # 处理BM25检索结果
            for doc_id, score in bm25_results:
                if doc_id < len(self.all_documents):
                    doc = self.all_documents[doc_id]
                    content_hash = hash(doc.page_content)
                    if content_hash in hybrid_scores:
                        stored_doc, stored_score = hybrid_scores[content_hash]
                        hybrid_scores[content_hash] = (stored_doc, stored_score + score * 0.4)
                    elif content_hash in vector_docs_map:  # 确保文档在向量检索结果中
                        hybrid_scores[content_hash] = (vector_docs_map[content_hash], score * 0.4)
            
            # 按混合分数排序
            sorted_docs = sorted(hybrid_scores.values(), key=lambda x: x[1], reverse=True)[:actual_top_k]
        else:
            # 回退到仅使用向量检索
            sorted_docs = [(doc, 1.0) for doc in related_docs]
        
        # 提取文档内容
        docs_only = [doc for doc, _ in sorted_docs]
        related_content = get_related_content(docs_only)
        
        return related_content

        
    
    def _format_results(self, reranked_docs) -> dict[str, Any]:
        unique_segments = {}
        segments = []
    
        for doc, score in reranked_docs:
        # 生成唯一标识（使用元数据中的路径+页码或内容哈希）
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}" or str(hash(doc.page_content))
        
            if doc_id not in unique_segments:
                segments.append({
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
                })
                unique_segments[doc_id] = True
    
        return {
            "text": "\n".join([seg["content"] for seg in segments]),
            "segments": segments
        }



class LarkSuiteOnlineRAG(BasicRAG):
    URL: List[str]
    vs_path: str
    load_files: List[str] = []
    top_k: int
    embedding: object
    llm: object
    conversation_with_summary: object
    init: bool = True
    vector_store: object = None
    bm25: object = None
    all_documents: List = None
    hnsw_wrapper: object = None  # HNSW包装器

    def __init__(self, URL: List[str], vs_path: str, embeddings: object,
                       init: bool = True, llm = None, use_hnsw: bool = False,
                       concurrent_workers: int = 8, top_k: int = 1, clear_existing: bool = True) -> None:
        self.use_hnsw = use_hnsw
        self.concurrent_workers = concurrent_workers
        self.previous_query = None  # 存储用户的上一次提问
        self.last_query_type = None  # 存储上一次查询的类型
        
        # 初始化向量存储为None
        self.vector_store = None
        
        if init:
            # 保存原始vs_path，用于后续清理
            original_vs_path = vs_path
            
            # 初始化向量库，这会返回更新后的vs_path
            vs_path, loaded_files = init_larksuite_vector_store(url=URL,
                                                               embeddings=embeddings,
                                                               vs_path=vs_path,
                                                               clear_existing=clear_existing)
        else:
            # 使用传入的vs_path参数，而不是硬编码的全局变量
            loaded_files = []

        self.load_files = loaded_files
        self.vs_path = vs_path
        self.URL = URL
        self.embeddings = embeddings
        self.top_k = top_k
        self.llm = llm
        
        # 初始化HNSW索引（如果启用）
        if self.use_hnsw and HNSW_AVAILABLE:
            try:
                self.hnsw_wrapper = HNSWRAGWrapper(
                    filepath=URL,
                    vs_path=vs_path,
                    embeddings=embeddings,
                    init=init,
                    llm=llm,
                    top_k=top_k,
                    clear_existing=clear_existing,
                    concurrent_workers=self.concurrent_workers
                )
                #logger.info("HNSW索引已启用")
            except Exception as e:
                #logger.error(f"HNSW索引初始化失败: {e}")
                self.hnsw_wrapper = None
                self.use_hnsw = False
        else:
            self.hnsw_wrapper = None
        
        # 初始化向量存储实例一次，避免每次查询都重新加载
        self.vector_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
        
        # 初始化重排序器
        # self.reranker = CrossEncoder(model_name_or_path='./ReRankers', device='cpu')
        
        # 预加载所有文档并初始化BM25
        self._preload_documents()
    
    def hybrid_retrieval(self, similarity_results, mmr_results, top_k) -> list[tuple]:
        score_map = {}
        for doc, score in similarity_results:
            content_hash = hash(doc.page_content)
            score_map[content_hash] = (doc, score * 0.7)  # 
    
        # 处理MMR结果（假设mmr_results是Document列表）
        for doc in mmr_results:
            content_hash = hash(doc.page_content)
            mmr_score = doc.metadata.get('relevance', 0) * 0.3
            if content_hash in score_map:
                stored_doc, stored_score = score_map[content_hash]
                score_map[content_hash] = (stored_doc, stored_score + mmr_score)
            else:
                score_map[content_hash] = (doc, mmr_score)
    
        # 按分数排序并返回文档对象
        sorted_docs = sorted(score_map.values(), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in sorted_docs[:top_k]]

    
    def query_knowledge(self, query: str) -> LiteralString:
        # 如果启用了HNSW索引且可用，则使用HNSW检索
        if self.use_hnsw and self.hnsw_wrapper:
            try:
                return self.hnsw_wrapper.query_knowledge(query)
            except Exception as e:
                print(f"HNSW检索失败，回退到Chroma检索: {e}")
        
        # vector_store = FAISS.load_local(self.vs_path, self.embeddings)
        vector_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
        actual_top_k = min(self.top_k, vector_store._collection.count())
        # 添加元数据过滤条件
        # similarity_docs = vector_store.similarity_search_with_score(query, k=self.top_k)
        similarity_docs = vector_store.similarity_search_with_score(query, k=actual_top_k)
        mmr_docs = vector_store.max_marginal_relevance_search(query, k=actual_top_k)
        combined_docs = self.hybrid_retrieval(similarity_docs, mmr_docs, actual_top_k)
        return get_related_content([doc for doc, _ in combined_docs])
    
    def query_knowledge_hnsw_with_scores(self, query: str):
        """使用HNSW索引查询并返回带分数的结果"""
        if self.use_hnsw and self.hnsw_wrapper:
            try:
                return self.hnsw_wrapper.query_knowledge_with_scores(query)
            except Exception as e:
                #logger.error(f"HNSW检索失败: {e}")
                return []
        else:
            #logger.warning("HNSW索引未启用或不可用")
            return []
    
    def _preload_documents(self):
        # 预加载所有文档内容用于BM25初始化
        self.all_documents = []
        if self.vector_store:
            # 获取所有文档内容
            results = self.vector_store.get()
            if results and 'documents' in results:
                # 构建完整的Document对象
                for i, content in enumerate(results['documents']):
                    doc = Document(page_content=content)
                    if 'metadatas' in results and i < len(results['metadatas']):
                        doc.metadata = results['metadatas'][i]
                    self.all_documents.append(doc)
                
                # 初始化BM25模型
                corpus = [doc.page_content for doc in self.all_documents]
                self.bm25 = BM25(corpus)
    
    def query_knowledge_BM25(self, query: str):
        # 检查当前查询是否为"自己已有知识回答以上问题"
        if query == "自己已有知识回答以上问题":
            # 对于此类查询，不召回上下文，返回空字符串
            self.last_query_type = "knowledge_only"
            return ""
        
        # 存储当前查询作为上一次提问
        self.previous_query = query
        self.last_query_type = "normal"
        
        # 如果启用了HNSW索引且可用，则使用HNSW检索
        if self.use_hnsw and self.hnsw_wrapper:
            try:
                return self.hnsw_wrapper.query_knowledge(query)
            except Exception as e:
                print(f"HNSW检索失败，回退到Chroma检索: {e}")
        
        # 使用预初始化的向量存储实例
        actual_top_k = min(self.top_k, self.vector_store._collection.count())
        related_docs_with_score = self.vector_store.similarity_search_with_score(query, k=actual_top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        
        # 使用预初始化的BM25模型
        if self.bm25 and self.all_documents:
            # 执行BM25检索
            bm25_results = self.bm25.search(query, top_n=len(self.all_documents))
            
            # 混合检索结果 (向量权重0.6, BM25权重0.4)
            # 调整权重以提高检索质量
            hybrid_scores = {}
            
            # 为向量检索结果创建快速查找映射
            vector_docs_map = {hash(doc.page_content): doc for doc in related_docs}
            
            # 处理向量检索结果
            for doc in related_docs:
                content_hash = hash(doc.page_content)
                vector_score = doc.metadata.get("score", 0)
                hybrid_scores[content_hash] = (doc, vector_score * 0.6)
            
            # 处理BM25检索结果
            for doc_id, score in bm25_results:
                if doc_id < len(self.all_documents):
                    doc = self.all_documents[doc_id]
                    content_hash = hash(doc.page_content)
                    if content_hash in hybrid_scores:
                        stored_doc, stored_score = hybrid_scores[content_hash]
                        hybrid_scores[content_hash] = (stored_doc, stored_score + score * 0.4)
                    elif content_hash in vector_docs_map:  # 确保文档在向量检索结果中
                        hybrid_scores[content_hash] = (vector_docs_map[content_hash], score * 0.4)
            
            # 按混合分数排序
            sorted_docs = sorted(hybrid_scores.values(), key=lambda x: x[1], reverse=True)[:actual_top_k]
        else:
            # 回退到仅使用向量检索
            sorted_docs = [(doc, 1.0) for doc in related_docs]
        
        # 提取文档内容
        docs_only = [doc for doc, _ in sorted_docs]
        related_content = get_related_content(docs_only)
        
        return related_content

        
    
    def _format_results(self, reranked_docs) -> dict[str, Any]:
        unique_segments = {}
        segments = []
    
        for doc, score in reranked_docs:
        # 生成唯一标识（使用元数据中的路径+页码或内容哈希）
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}" or str(hash(doc.page_content))
        
            if doc_id not in unique_segments:
                segments.append({
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
                })
                unique_segments[doc_id] = True
    
        return {
            "text": "\n".join([seg["content"] for seg in segments]),
            "segments": segments
        }

    