from typing import List
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.TextSplitter import ChineseTextSplitter, WindowedTextSplitter
from langchain_chroma import Chroma
import datetime
import torch
from tqdm import tqdm
from models.config import *
from sentence_transformers import CrossEncoder
from typing import *

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
        return docs
    except Exception as e:
        print(f"文件加载失败: {str(e)}")
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
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")


def load_file(filepath: str) -> list[Document]:
    if filepath.lower().endswith(".md"):
        loader = UnstructuredMarkdownLoader(filepath, mode="elements")
        textsplitter = WindowedTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = loader.load()
        # 新增元数据过滤
        # docs = filter_complex_metadata(docs)
        # 保留原有的特定字段处理（可选）
        # for doc in docs:
        #     for key in ['language', 'category', 'source']:
        #         if key in doc.metadata and isinstance(doc.metadata[key], list):
        #             doc.metadata[key] = ','.join(doc.metadata[key])
        docs = filter_complex_metadata(docs)
        docs = textsplitter.split_documents(docs)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
        # 添加PDF的元数据过滤
        docs = filter_complex_metadata(docs)
    else:
        docs = load_txt_file(filepath)
        # 添加文本文件的元数据过滤
        docs = filter_complex_metadata(docs)
    return docs

def get_related_content(related_docs: Any) -> LiteralString:
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content)
    return "\n".join(related_content)

def get_docs_with_score(docs_with_score: Any) -> List:
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs

def init_knowledge_vector_store(filepath: str | List[str],
                                vs_path: str | os.PathLike = None,
                                embeddings: object = None) -> (tuple[str | os.PathLike, list] | tuple[None, list] | None):
    loaded_files = []
    failed_files = []
    # 单个文件
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print(f"{filepath} 路径不存在")
            return None
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            try:
                docs = load_file(filepath)
                print(f"{file} 已成功加载")
                loaded_files.append(filepath)
            except Exception as e:
                print(e)
                print(f"{file} 未能成功加载")
                return None
        elif os.path.isdir(filepath):
            docs = []
            for file in tqdm(os.listdir(filepath), desc="加载文件"):
                fullfilepath = os.path.join(filepath, file)

                try:
                    docs += load_file(fullfilepath)
                    loaded_files.append(fullfilepath)
                except Exception as e:
                    print(e)
                    failed_files.append(file)

            if len(failed_files) > 0:
                print("以下文件未能成功加载：")
                for file in failed_files:
                    print(file,end="\n")
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
        print("文件加载完毕，正在生成向量库")
        if vs_path and os.path.isdir(vs_path):
            # vector_store = FAISS.load_local(vs_path, embeddings)
            vector_store = Chroma(persist_directory=vs_path, embedding_function=embeddings)
            vector_store.add_documents(docs)
            torch_gc()
        else:
            if not vs_path:
                vs_path = os.path.join(vs_path,
                                       f"""CHROMA{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
            vector_store = Chroma.from_documents(docs, embeddings, persist_directory=vs_path)
            torch_gc()

        # vector_store.save_local(vs_path)
        print("向量生成成功")
        return vs_path, loaded_files
    else:
        print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
        return None, loaded_files

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

    def __init__(self, filepath: str, vs_path: str, embeddings: object,
                       init: bool = True, llm = None) -> None:
        if init:
            vs_path, loaded_files = init_knowledge_vector_store(filepath=filepath,
                                                                vs_path=VS_PATH_EXCEL,
                                                                embeddings=embeddings)
        else:
            vs_path = VS_PATH_EXCEL
            loaded_files = []


        self.load_files = loaded_files
        self.vs_path = vs_path
        self.filepath = filepath
        self.embeddings = embeddings
        self.top_k = VECTOR_SEARCH_TOP_K_EXCEL
        # self.llm = CustomLLM()
        self.llm = ChatOpenAI(model="qwq:32b", base_url='http://10.81.38.110:11434/v1', max_completion_tokens=32768,
                         temperature=0, top_p=0.9, streaming=True)
        self.llm = llm
        
        # self.conversation_with_summary = ConversationChain(llm=self.llm,
        #                                                memory=ConversationSummaryBufferMemory(llm=self.llm,
        #                                                                                       max_token_limit=40),
        #                                                verbose=True)
        self.reranker = CrossEncoder(model_name_or_path='./ReRankers', device='cpu')
    
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

    # def query_knowledge_multi(self, query: str, exclude_segments: set = None) -> dict[str, Any]:
    #     # vector_store = FAISS.load_local(self.vs_path, self.embeddings)
    #     vector_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
    #     actual_top_k = min(self.top_k, vector_store._collection.count())
    #     # 添加元数据过滤条件
    #     filter_dict = {"id": {"$nin": list(exclude_segments)}} if exclude_segments else None
    #     # similarity_docs = vector_store.similarity_search_with_score(query, k=self.top_k)
    #     similarity_docs = vector_store.similarity_search_with_score(query, k=actual_top_k, filter=filter_dict)
    #     mmr_docs = vector_store.max_marginal_relevance_search(query, k=actual_top_k)
    #     combined_docs = self.hybrid_retrieval(similarity_docs, mmr_docs, actual_top_k)

    #     query_doc_pairs = [(query, doc.page_content) for doc, _ in combined_docs]
    #     scores = self.reranker.predict(query_doc_pairs)
    #     reranked = sorted(zip([doc for doc, _ in combined_docs], scores), 
    #                      key=lambda x: x[1], reverse=True)[:5]  # 取前5个
        

    #     # related_docs = get_docs_with_score(combined_docs)
    #     # vector_store.chunk_size = CHUNK_SIZE
    #     # related_docs_with_score = vector_store.similarity_search_with_score(query, k = self.top_k)
    #     # related_docs = get_docs_with_score(related_docs_with_score)
    #     # related_content = get_related_content(related_docs)
    #     # return related_content
    #     # return get_related_content([doc for doc, _ in reranked])
    #     return self._format_results(reranked)
    
    def query_knowledge(self, query: str) -> LiteralString:
        # vector_store = FAISS.load_local(self.vs_path, self.embeddings)
        vector_store = Chroma(persist_directory=self.vs_path, embedding_function=self.embeddings)
        actual_top_k = min(self.top_k, vector_store._collection.count())
        # 添加元数据过滤条件
        # similarity_docs = vector_store.similarity_search_with_score(query, k=self.top_k)
        similarity_docs = vector_store.similarity_search_with_score(query, k=actual_top_k)
        mmr_docs = vector_store.max_marginal_relevance_search(query, k=actual_top_k)
        combined_docs = self.hybrid_retrieval(similarity_docs, mmr_docs, actual_top_k)

        query_doc_pairs = [(query, doc.page_content) for doc, _ in combined_docs]
        scores = self.reranker.predict(query_doc_pairs)
        reranked = sorted(zip([doc for doc, _ in combined_docs], scores), 
                         key=lambda x: x[1], reverse=True)[:1]  #
        

        # related_docs = get_docs_with_score(combined_docs)
        # vector_store.chunk_size = CHUNK_SIZE
        # related_docs_with_score = vector_store.similarity_search_with_score(query, k = self.top_k)
        # related_docs = get_docs_with_score(related_docs_with_score)
        # related_content = get_related_content(related_docs)
        # return related_content
        return get_related_content([doc for doc, _ in reranked])
    
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

    # def get_knowledge_based_answer(self, query: str):
    #     related_content = self.query_knowledge(query)
    #     print("召回文本段:", related_content)
    #     # prompt = PromptTemplate(
    #     #     input_variables=["context","question"],
    #     #     template=PROMPT_TEMPLATE,
    #     # )
    #     promrt = ChatPromptTemplate.from_messages([
    #         ("system", "你是一个正在跟某个人类对话的机器人."),
    #         ("human", "{context}"),
    #         ("human", "{question}")
    #     ])
    #     pmt = promrt.format(context=related_content,
    #                         question=query)

    #     # answer=self.conversation_with_summary.predict(input=pmt)
    #         # if run_manager:
    #     answer = self.llm.invoke(pmt)
    #     return answer
