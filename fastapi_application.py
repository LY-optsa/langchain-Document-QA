#coding=utf-8
'''
Application build by using FastAPI
Update:2025-12-10 19:47
Updateing:
1. Add file upload and delete
2. Add vector store update
author:LiYu
'''

from models.config import *
from models import RAG, LarkSuiteOnlineRAG
from models import save_conversation, get_conversation_history, delete_conversation_history, QueryRequest, QueryResponse, DeleteConversationRequest
from models import update_knowledge_vector_store, update_larksuite_vector_store
#from models.SearchToolAgent import SearchAgent
from models.SearchToolAgent import SearchAgent
# Import MCP tools
# from models.mcp_tools import mcp as mcp_server, run_mcp
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import AsyncIteratorCallbackHandler
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import os
from typing import *
import asyncio
import json
import uuid
import sqlite3
import concurrent.futures
#import logging
import shutil
import threading
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

lark_suite_doc_url = [
            "https://open.feishu.cn/open-apis/sheets/v3/spreadsheets/JtIUs8sIUhIVe3tYSJMcUZiLncb",
            "https://open.feishu.cn/open-apis/sheets/v3/spreadsheets/IYuksoXvJhjoAatRh0DcE5BInPr",
            "https://open.feishu.cn/open-apis/sheets/v3/spreadsheets/XL39s5VVUhdRPxtfK8BcHg8pnjh"
      ]

# 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("document_qa.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("DocumentQA")

# 创建线程池执行器
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="DocumentQA")

# 定义在线程池中运行函数的装饰器
def run_in_threadpool(func):
    async def wrapper(*args, **kwargs):
        return await asyncio.get_event_loop().run_in_executor(
            thread_pool, lambda: func(*args, **kwargs)
        )
    return wrapper

app = FastAPI(title="Document QA API for FastAPI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建数据库目录
db_dir = os.path.join(os.path.dirname(__file__), "databases")
os.makedirs(db_dir, exist_ok=True)

# 数据库路径
EXCEL_DB_PATH = os.path.join(db_dir, "excel_conversations.db")
PDF_DB_PATH = os.path.join(db_dir, "pdf_conversations.db")
# llm = ChatOllama(model='qwen2.5:14b', base_url='http://localhost:11434/v1', 
#                  temperature=0.0, top_p=0.95, frequency_penalty=0.0)
api_key = os.getenv("OpenAI_API_Key")
llm = ChatOpenAI(model='Qwen3-235B-A22B-Instruct-2507', api_key=api_key, base_url='http://localhost:11434/v1', 
                 temperature=0.0, top_p=0.95, frequency_penalty=0.0)
embeddings = OllamaEmbeddings(model='quentinz/bge-large-zh-v1.5:latest', base_url='http://localhost:11434')


rag_instances: Dict[str, RAG] = {}
def init_database(db_path):
    """初始化对话历史数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 创建对话历史表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        user_question TEXT NOT NULL,
        assistant_answer TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # 创建索引以加快查询速度
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversation_history(conversation_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversation_history(timestamp)')
    conn.commit()
    conn.close()


# 创建异步版本的数据库操作函数
init_database_async = run_in_threadpool(init_database)
save_conversation_async = run_in_threadpool(save_conversation)
get_conversation_history_async = run_in_threadpool(get_conversation_history)
delete_conversation_history_async = run_in_threadpool(delete_conversation_history)

# 为RAG查询创建异步版本
def query_knowledge_sync(rag_instance, question):
    """同步版本的RAG查询"""
    try:
        return rag_instance.query_knowledge_BM25(question)
    except Exception as e:
       # logger.error(f"RAG查询错误: {str(e)}")
        return ""

query_knowledge_async = run_in_threadpool(query_knowledge_sync)

def initialize_rag_instances():
    global rag_instances
    # 初始化数据库
    init_database(EXCEL_DB_PATH)
    init_database(PDF_DB_PATH)
    
    # 初始化Excel RAG实例
    # if os.path.exists(DATA_DIR_EXCEL):
    #     rag_instances["excel"] = RAG(
    #         filepath=DATA_DIR_EXCEL,
    #         vs_path=VS_PATH_EXCEL,
    #         embeddings=embeddings,
    #         init=True if not os.path.exists(VS_PATH_EXCEL) else False,
    #         llm=llm,
    #         use_hnsw=False,
    #         top_k=VECTOR_SEARCH_TOP_K_EXCEL
    #     )
    if len(lark_suite_doc_url) > 0:
        rag_instances["excel"] = LarkSuiteOnlineRAG(
            URL=lark_suite_doc_url,
            vs_path=VS_PATH_LARKSUITE,
            embeddings=embeddings,
            init=True if not os.path.exists(VS_PATH_LARKSUITE) else False,
            llm=llm,
            top_k=VECTOR_SEARCH_TOP_K_EXCEL
        )
    
    # 初始化PDF RAG实例
    if os.path.exists(DATA_DIR_PDF):
        rag_instances["pdf"] = RAG(
            filepath=DATA_DIR_PDF,
            vs_path=VS_PATH_PDF,
            embeddings=embeddings,
            init=True if not os.path.exists(VS_PATH_PDF) else False,
            llm=llm,
            use_hnsw=False,
            top_k=VECTOR_SEARCH_TOP_K_PDF
        )


def update_rag_instances_excel():
    results = update_larksuite_vector_store(embedding=embeddings, url=lark_suite_doc_url, vs_path=VS_PATH_LARKSUITE)
    return results
    
def update_rag_instances_pdf():
    results = update_knowledge_vector_store(embeddings, DATA_DIR_PDF, VS_PATH_PDF, delete_existing=False)
    return results
    

def is_unable_to_answer(answer: str) -> bool:
    """检查回答是否包含无法回答的关键词"""
    unable_keywords = [
        "无法回答", "没有相关信息", "不了解", "不知道", "无法提供", 
        "没有足够信息", "无法确定", "无法给出", "无法判断", "没有找到"
    ]
    # 将回答转为小写进行比较
    answer_lower = answer.lower()
    # 检查是否包含任何无法回答的关键词
    return any(keyword.lower() in answer_lower for keyword in unable_keywords)

async def send_message(question: str, rag_instance: RAG, conversation_id: str, db_path: str) -> AsyncIterable[str]:
    # logger.info(f"开始处理请求: conversation_id={conversation_id}")
    
    # 检查是否需要使用外部知识
    use_external_knowledge = "外部知识" in question
    # 检查是否同时需要使用知识库和外部知识
    use_rag_and_external_knowledge = "外部知识" in question and "知识库" in question
    
    if use_rag_and_external_knowledge:
        # logger.info(f"同时使用知识库和外部知识: conversation_id={conversation_id}")
        # 异步获取对话历史
        history = await get_conversation_history_async(db_path, conversation_id)
        
        # 确定要搜索的查询内容
        search_query = question  # 默认使用当前问题
        if history and len(history) > 0:
            # 如果有历史对话，使用最后一个用户提问作为搜索查询
            last_user_question = history[-1][0]  # history格式为[(user_q, assistant_a), ...]
            search_query = last_user_question
            #logger.info(f"使用历史提问作为搜索查询: {search_query}")
        
        # 同时获取上下文信息
        context = await query_knowledge_async(rag_instance, question)
        #logger.info(f"同时获取上下文信息完成: conversation_id={conversation_id}")
        
        # 创建增强的搜索查询，包含上下文信息
        enhanced_query = f"问题: {search_query}\n\n相关上下文信息: {context}"
        
        # 首先发送conversation_id
        first_data = {"conversation_id": conversation_id, "answer": ""}
        yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
        
        # 收集完整的回答
        full_answer = ""
        
        # 使用异步流式搜索
        async for chunk in SearchAgent.astream_query(enhanced_query):
            if chunk:
                full_answer += chunk
                # 使用标准SSE格式
                data = {"conversation_id": conversation_id, "answer": chunk}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        # 保存对话历史
        if full_answer:
            await save_conversation_async(db_path, conversation_id, question, full_answer)
            #logger.info(f"知识库和外部知识结合搜索完成并保存历史: conversation_id={conversation_id}")
        return
    
    if use_external_knowledge:
        #logger.info(f"使用外部知识搜索: conversation_id={conversation_id}")
        # 异步获取对话历史，以查找历史提问
        history = await get_conversation_history_async(db_path, conversation_id)
        
        # 确定要搜索的查询内容
        search_query = question  # 默认使用当前问题
        if history and len(history) > 0:
            # 如果有历史对话，使用最后一个用户提问作为搜索查询
            last_user_question = history[-1][0]  # history格式为[(user_q, assistant_a), ...]
            search_query = last_user_question
            #logger.info(f"使用历史提问作为搜索查询: {search_query}")
        
        # 首先发送conversation_id
        first_data = {"conversation_id": conversation_id, "answer": ""}
        yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
        
        # 收集完整的回答
        full_answer = ""
        
        # 使用异步流式搜索
        async for chunk in SearchAgent.astream_query(search_query):
            if chunk:
                full_answer += chunk
                # 使用标准SSE格式
                data = {"conversation_id": conversation_id, "answer": chunk}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        # 保存对话历史
        if full_answer:
            await save_conversation_async(db_path, conversation_id, question, full_answer)
            #logger.info(f"外部知识搜索完成并保存历史: conversation_id={conversation_id}")
        return
    
    # 正常的RAG处理流程
    # 异步获取对话历史
    history = await get_conversation_history_async(db_path, conversation_id)
    
    # 创建回调处理器
    callback = AsyncIteratorCallbackHandler()
    
    # 创建带有回调处理器的LLM实例
    streaming_llm = ChatOpenAI(
        model='Qwen3-235B-A22B-Instruct-2507',
        api_key=api_key,
        base_url='http://10.111.32.151:3001/v1',
        temperature=0.0,
        top_p=0.95,
        frequency_penalty=0.0,
        streaming=True,
        callbacks=[callback]
    )
    
    # 异步使用RAG查询知识，获取上下文
    context = await query_knowledge_async(rag_instance, question)
    
    # 格式化对话历史
    chat_history_str = ""
    for user_q, assistant_a in history:
        chat_history_str += f"用户: {user_q}\n助手: {assistant_a}\n\n"
    
    # 添加系统消息
    messages = []
    messages.append(SystemMessage(content=TEMPLATE.replace("{chat_history}", chat_history_str).replace("{question}", question).replace("{context}", context)))
    
    # 添加历史对话
    for user_q, assistant_a in history:
        messages.append(HumanMessage(content=user_q))
        messages.append(AIMessage(content=assistant_a))
    
    # 添加当前问题
    messages.append(HumanMessage(content=question))

    # 创建生成任务并在后台运行
    task = asyncio.create_task(
        streaming_llm.agenerate(messages=[messages])
    )

    try:
        # 首先发送conversation_id
        first_data = {"conversation_id": conversation_id, "answer": ""}
        yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
        
        # 收集完整的回答
        full_answer = ""
        
        # 从回调迭代器中获取并流式传输每个token
        async for token in callback.aiter():
            if token:
                full_answer += token
                # 使用标准SSE格式
                data = {"conversation_id": conversation_id, "answer": token}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        # 检查回答是否包含无法回答的关键词
        if is_unable_to_answer(full_answer):
           # logger.info(f"检测到无法回答的内容，切换到外部搜索: conversation_id={conversation_id}")
            # 发送一个提示，表明我们正在使用外部搜索
            search_notice = "[正在使用外部搜索补充信息...]"
            yield f"data: {json.dumps({"conversation_id": conversation_id, "answer": search_notice}, ensure_ascii=False)}\n\n"
            
            # 使用外部搜索获取答案
            external_full_answer = ""
            async for chunk in SearchAgent.astream_query(question):
                if chunk:
                    external_full_answer += chunk
                    # 使用标准SSE格式
                    data = {"conversation_id": conversation_id, "answer": chunk}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            
            # 保存对话历史（使用外部搜索的结果）
            if external_full_answer:
                await save_conversation_async(db_path, conversation_id, question, external_full_answer)
                #logger.info(f"外部搜索完成并保存历史: conversation_id={conversation_id}")
        else:
            # 异步保存对话历史（使用原始RAG结果）
            if full_answer:
                await save_conversation_async(db_path, conversation_id, question, full_answer)
                #logger.info(f"请求处理完成并保存历史: conversation_id={conversation_id}")
            
    except Exception as e:
        error_data = {"conversation_id": conversation_id, "error": str(e)}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        #logger.error(f"处理请求时发生错误: conversation_id={conversation_id}, error={str(e)}")
    finally:
        # 确保任务完成
        callback.done.set()
        try:
            await task
        except Exception as e:
            information = f"任务执行异常: conversation_id={conversation_id}, error={str(e)}"
            #logger.error(f"任务执行异常: conversation_id={conversation_id}, error={str(e)}")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库和RAG实例"""
    initialize_rag_instances()
    #logger.info("数据库和RAG实例初始化完成")

@app.post("/upload/excel")
async def upload_excel_files(files: List[UploadFile] = File(...)):
    path = DATA_DIR_EXCEL
    for file in files:
        with open(os.path.join(path, file.filename), "wb") as f:
            f.write(await file.read())
        f.close()
    return {"filenames": [file.filename for file in files]}

@app.post("/upload/pdf")
async def upload_pdf_files(files: List[UploadFile] = File(...)):
    path = DATA_DIR_PDF
    for file in files:
        with open(os.path.join(path, file.filename), "wb") as f:
            f.write(await file.read())
        f.close()
    return {"filenames": [file.filename for file in files]}

@app.post("/update/excel")
async def update_excel_rag():
    results = update_rag_instances_excel()
    data = {"add": results["num_added"], "update": results["num_updated"], "skip": results["num_skipped"], "delete": results["num_deleted"]}
    return {"message": "飞书RAG实例更新完成", "data":data}

@app.post("/update/pdf")
async def update_pdf_rag():
    results = update_rag_instances_pdf()
    data = {"add": results["num_added"], "update": results["num_updated"], "skip": results["num_skipped"], "delete": results["num_deleted"]}
    return {"message":  "更新完成", "data":data}

@app.post("/query/excel")
async def query_excel(request: QueryRequest):
    # 获取全局变量
    global rag_instances
    
    question = request.question
    conversation_id = request.conversation_id or str(uuid.uuid4())
    top_k = request.top_k or VECTOR_SEARCH_TOP_K_EXCEL
    model = request.model or "GW_DeepSeek-R1"
    temperature = request.temperature or 0.5
    top_p = request.top_p or 0.95
    llm_top_k = request.llm_top_k or 10
    stream = request.stream or False
    
    # 检查是否需要使用外部知识
    use_external_knowledge = "外部知识" in question
    # 检查是否同时需要使用知识库和外部知识
    use_rag_and_external_knowledge = "外部知识" in question and "知识库" in question
    
    if use_rag_and_external_knowledge:
        #logger.info(f"同时使用知识库和外部知识搜索Excel查询: conversation_id={conversation_id}")
        # 获取对话历史以查找历史提问
        history = await get_conversation_history_async(EXCEL_DB_PATH, conversation_id)
        
        # 确定要搜索的查询内容
        search_query = question  # 默认使用当前问题
        if history and len(history) > 0:
            # 如果有历史对话，使用最后一个用户提问作为搜索查询
            last_user_question = history[-1][0]  # history格式为[(user_q, assistant_a), ...]
            search_query = last_user_question
            #logger.info(f"使用历史提问作为搜索查询: {search_query}")
        
        # 获取RAG实例
        rag = rag_instances["excel"]
        
        # 同时获取上下文信息
        context = await query_knowledge_async(rag, question)
        #logger.info(f"同时获取上下文信息完成: conversation_id={conversation_id}")
        
        # 创建增强的搜索查询，包含上下文信息
        enhanced_query = f"问题: {search_query}\n\n相关上下文信息: {context}"
        
        if stream:
            # 流式输出使用SearchAgent
            async def external_knowledge_generator():
                # 首先发送conversation_id
                first_data = {"conversation_id": conversation_id, "answer": ""}
                yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
                
                # 收集完整的回答
                full_answer = ""
                
                # 使用异步流式搜索
                async for chunk in SearchAgent.astream_query(enhanced_query):
                    if chunk:
                        full_answer += chunk
                        data = {"conversation_id": conversation_id, "answer": chunk}
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # 保存对话历史
                if full_answer:
                    await save_conversation_async(EXCEL_DB_PATH, conversation_id, question, full_answer)
            
            return StreamingResponse(external_knowledge_generator(), media_type="text/event-stream")
        else:
            try:
                # 非流式输出使用SearchAgent
                result = await SearchAgent.aquery(enhanced_query)
                answer = result.get("output", "")
                
                # 保存对话历史
                await save_conversation_async(EXCEL_DB_PATH, conversation_id, question, answer)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=answer).dict())
            except Exception as e:
                #logger.error(f"处理知识库和外部知识结合搜索时发生错误: conversation_id={conversation_id}, error={str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "conversation_id": conversation_id}
                )
    
    if use_external_knowledge:
        #logger.info(f"使用外部知识搜索Excel查询: conversation_id={conversation_id}")
        # 获取对话历史以查找历史提问
        history = await get_conversation_history_async(EXCEL_DB_PATH, conversation_id)
        
        # 确定要搜索的查询内容
        search_query = question  # 默认使用当前问题
        if history and len(history) > 0:
            # 如果有历史对话，使用最后一个用户提问作为搜索查询
            last_user_question = history[-1][0]  # history格式为[(user_q, assistant_a), ...]
            search_query = last_user_question
            #logger.info(f"使用历史提问作为搜索查询: {search_query}")
        
        if stream:
            # 流式输出使用SearchAgent
            async def external_knowledge_generator():
                # 首先发送conversation_id
                first_data = {"conversation_id": conversation_id, "answer": ""}
                yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
                
                # 收集完整的回答
                full_answer = ""
                
                # 使用异步流式搜索
                async for chunk in SearchAgent.astream_query(search_query):
                    if chunk:
                        full_answer += chunk
                        data = {"conversation_id": conversation_id, "answer": chunk}
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # 保存对话历史
                if full_answer:
                    await save_conversation_async(EXCEL_DB_PATH, conversation_id, question, full_answer)
            
            return StreamingResponse(external_knowledge_generator(), media_type="text/event-stream")
        else:
            try:
                # 非流式输出使用SearchAgent
                result = await SearchAgent.aquery(search_query)
                answer = result.get("output", "")
                
                # 保存对话历史
                await save_conversation_async(EXCEL_DB_PATH, conversation_id, question, answer)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=answer).dict())
            except Exception as e:
                #logger.error(f"处理外部知识搜索时发生错误: conversation_id={conversation_id}, error={str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "conversation_id": conversation_id}
                )
    
    # 获取RAG实例
    rag = rag_instances["excel"]
    
    if stream:
        # 流式输出也使用RAG，并传递conversation_id和数据库路径
        generator = send_message(question, rag, conversation_id, EXCEL_DB_PATH)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        try:
            # 异步获取对话历史
            history = await get_conversation_history_async(EXCEL_DB_PATH, conversation_id)
            
            # 异步使用RAG查询知识，获取上下文
            context = await query_knowledge_async(rag, question)
            
            # 构建完整的消息列表，包括历史对话
            messages = []
            
            # 格式化对话历史
            chat_history_str = ""
            for user_q, assistant_a in history:
                chat_history_str += f"用户: {user_q}\n助手: {assistant_a}\n\n"
            
            messages.append(SystemMessage(content=TEMPLATE.replace("{chat_history}", chat_history_str).replace("{question}", question).replace("{context}", context)))
            
            # 添加历史对话
            for user_q, assistant_a in history:
                messages.append(HumanMessage(content=user_q))
                messages.append(AIMessage(content=assistant_a))
            
            # 添加当前问题
            messages.append(HumanMessage(content=question))
            
            # 获取回答
            result = llm.invoke(messages).content
            
            # 检查回答是否包含无法回答的关键词
            if is_unable_to_answer(result):
                #logger.info(f"检测到无法回答的内容，切换到外部搜索: conversation_id={conversation_id}")
                # 使用外部搜索获取答案
                external_result = await SearchAgent.aquery(question)
                external_answer = external_result.get("output", "")
                
                # 保存对话历史（使用外部搜索的结果）
                await save_conversation_async(EXCEL_DB_PATH, conversation_id, question, external_answer)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=external_answer).dict())
            else:
                # 异步保存对话历史（使用原始RAG结果）
                await save_conversation_async(EXCEL_DB_PATH, conversation_id, question, result)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=result).dict())
        except Exception as e:
            #logger.error(f"处理Excel查询时发生错误: conversation_id={conversation_id}, error={str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "conversation_id": conversation_id}
            )
        


@app.post("/query/pdf")
async def query_pdf(request: QueryRequest):
    # 获取全局变量
    global rag_instances
    
    question = request.question
    conversation_id = request.conversation_id or str(uuid.uuid4())
    top_k = request.top_k or VECTOR_SEARCH_TOP_K_PDF
    model = request.model or "GW_DeepSeek-R1"
    temperature = request.temperature or 0.5
    top_p = request.top_p or 0.95
    llm_top_k = request.llm_top_k or 10
    stream = request.stream or False
    # 检查是否需要使用外部知识
    use_external_knowledge = "外部知识" in question
    # 检查是否同时需要使用知识库和外部知识
    use_rag_and_external_knowledge = "外部知识" in question and "知识库" in question
    
    if use_rag_and_external_knowledge:
        #logger.info(f"同时使用知识库和外部知识搜索PDF查询: conversation_id={conversation_id}")
        # 获取对话历史以查找历史提问
        history = await get_conversation_history_async(PDF_DB_PATH, conversation_id)
        
        # 确定要搜索的查询内容
        search_query = question  # 默认使用当前问题
        if history and len(history) > 0:
            # 如果有历史对话，使用最后一个用户提问作为搜索查询
            last_user_question = history[-1][0]  # history格式为[(user_q, assistant_a), ...]
            search_query = last_user_question
            #logger.info(f"使用历史提问作为搜索查询: {search_query}")
        
        # 获取RAG实例
        rag = rag_instances["pdf"]
        
        # 同时获取上下文信息
        context = await query_knowledge_async(rag, question)
        #logger.info(f"同时获取上下文信息完成: conversation_id={conversation_id}")
        
        # 创建增强的搜索查询，包含上下文信息
        enhanced_query = f"问题: {search_query}\n\n相关上下文信息: {context}"
        
        if stream:
            # 流式输出使用SearchAgent
            async def external_knowledge_generator():
                # 首先发送conversation_id
                first_data = {"conversation_id": conversation_id, "answer": ""}
                yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
                
                # 收集完整的回答
                full_answer = ""
                
                # 使用异步流式搜索
                async for chunk in SearchAgent.astream_query(enhanced_query):
                    if chunk:
                        full_answer += chunk
                        data = {"conversation_id": conversation_id, "answer": chunk}
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # 保存对话历史
                if full_answer:
                    await save_conversation_async(PDF_DB_PATH, conversation_id, question, full_answer)
            
            return StreamingResponse(external_knowledge_generator(), media_type="text/event-stream")
        else:
            try:
                # 非流式输出使用SearchAgent
                result = await SearchAgent.aquery(enhanced_query)
                answer = result.get("output", "")
                
                # 保存对话历史
                await save_conversation_async(PDF_DB_PATH, conversation_id, question, answer)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=answer).dict())
            except Exception as e:
                #logger.error(f"处理知识库和外部知识结合搜索时发生错误: conversation_id={conversation_id}, error={str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "conversation_id": conversation_id}
                )
    
    if use_external_knowledge:
        #logger.info(f"使用外部知识搜索PDF查询: conversation_id={conversation_id}")
        # 获取对话历史以查找历史提问
        history = await get_conversation_history_async(PDF_DB_PATH, conversation_id)
        
        # 确定要搜索的查询内容
        search_query = question  # 默认使用当前问题
        if history and len(history) > 0:
            # 如果有历史对话，使用最后一个用户提问作为搜索查询
            last_user_question = history[-1][0]  # history格式为[(user_q, assistant_a), ...]
            search_query = last_user_question
            #logger.info(f"使用历史提问作为搜索查询: {search_query}")
        
        if stream:
            # 流式输出使用SearchAgent
            async def external_knowledge_generator():
                # 首先发送conversation_id
                first_data = {"conversation_id": conversation_id, "answer": ""}
                yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
                
                # 收集完整的回答
                full_answer = ""
                
                # 使用异步流式搜索
                async for chunk in SearchAgent.astream_query(search_query):
                    if chunk:
                        full_answer += chunk
                        data = {"conversation_id": conversation_id, "answer": chunk}
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # 保存对话历史
                if full_answer:
                    await save_conversation_async(PDF_DB_PATH, conversation_id, question, full_answer)
            
            return StreamingResponse(external_knowledge_generator(), media_type="text/event-stream")
        else:
            try:
                # 非流式输出使用SearchAgent
                result = await SearchAgent.aquery(search_query)
                answer = result.get("output", "")
                
                # 保存对话历史
                await save_conversation_async(PDF_DB_PATH, conversation_id, question, answer)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=answer).dict())
            except Exception as e:
                #logger.error(f"处理外部知识搜索时发生错误: conversation_id={conversation_id}, error={str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "conversation_id": conversation_id}
                )
    
    # 获取RAG实例
    rag = rag_instances["pdf"]
    
    if stream:
        # 流式输出也使用RAG，并传递conversation_id和数据库路径
        generator = send_message(question, rag, conversation_id, PDF_DB_PATH)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        try:
            # 异步获取对话历史
            history = await get_conversation_history_async(PDF_DB_PATH, conversation_id)
            
            # 异步使用RAG查询知识，获取上下文
            context = await query_knowledge_async(rag, question)
            
            # 构建完整的消息列表，包括历史对话
            messages = []
            
            # 格式化对话历史
            chat_history_str = ""
            for user_q, assistant_a in history:
                chat_history_str += f"用户: {user_q}\n助手: {assistant_a}\n\n"
            
            messages.append(SystemMessage(content=TEMPLATE.replace("{chat_history}", chat_history_str).replace("{question}", question).replace("{context}", context)))
            
            # 添加历史对话
            for user_q, assistant_a in history:
                messages.append(HumanMessage(content=user_q))
                messages.append(AIMessage(content=assistant_a))
            
            # 添加当前问题
            messages.append(HumanMessage(content=question))
            
            # 获取回答
            result = llm.invoke(messages).content
            
            # 检查回答是否包含无法回答的关键词
            if is_unable_to_answer(result):
               # logger.info(f"检测到无法回答的内容，切换到外部搜索: conversation_id={conversation_id}")
                # 使用外部搜索获取答案
                external_result = await SearchAgent.aquery(question)
                external_answer = external_result.get("output", "")
                
                # 保存对话历史（使用外部搜索的结果）
                await save_conversation_async(PDF_DB_PATH, conversation_id, question, external_answer)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=external_answer).dict())
            else:
                # 异步保存对话历史（使用原始RAG结果）
                await save_conversation_async(PDF_DB_PATH, conversation_id, question, result)
                
                return JSONResponse(content=QueryResponse(conversation_id=conversation_id, answer=result).dict())
        except Exception as e:
            #logger.error(f"处理PDF查询时发生错误: conversation_id={conversation_id}, error={str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "conversation_id": conversation_id}
            )

# 添加应用关闭时的事件处理，确保线程池正确关闭
@app.delete("/delete/conversation")
async def delete_conversation(request: DeleteConversationRequest):
    """删除指定对话ID的所有对话记录（同时从Excel和PDF数据库中删除）"""
    conversation_id = request.conversation_id
    
    try:
        # 异步删除Excel数据库中的对话记录
        deleted_count_excel = await delete_conversation_history_async(EXCEL_DB_PATH, conversation_id)
        
        # 异步删除PDF数据库中的对话记录
        deleted_count_pdf = await delete_conversation_history_async(PDF_DB_PATH, conversation_id)
        
        total_deleted = deleted_count_excel + deleted_count_pdf
        
       # logger.info(f"已删除对话记录: conversation_id={conversation_id}, excel_deleted={deleted_count_excel}, pdf_deleted={deleted_count_pdf}, total={total_deleted}")
        
        return JSONResponse(
            content={
                "message": "对话记录已成功删除",
                "conversation_id": conversation_id,
                "deleted_count": total_deleted
            }
        )
    except Exception as e:
       # logger.error(f"删除对话记录时发生错误: conversation_id={conversation_id}, error={str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "conversation_id": conversation_id}
        )
    
@app.delete("/delete_file")
async def delete_file(file_name: List[str]):
    try:
        for file in file_name:
            base_dir = DATA_DIR_EXCEL if file.endswith(".xlsx") or file.endswith(".xls") else DATA_DIR_PDF
            file_path = os.path.join(base_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                result = update_knowledge_vector_store(embedding=embeddings, filepath=DATA_DIR_PDF if file.endswith(".pdf") else DATA_DIR_EXCEL, 
                                                       vs_path=VS_PATH_PDF if file.endswith(".pdf") else VS_PATH_EXCEL, delete_existing=True)
                return JSONResponse(
                    content={
                        "message": "文件已成功删除",
                        "file": file,
                        "DELETED": result['num_deleted']
                    }
                )
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"文件不存在，无法删除: {file_path}"}
                )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "DocumentQA API"}

@app.get("/get_knowledge_excel")
async def get_knowledge_excel():
    path = DATA_DIR_EXCEL
    try:
        files = [file for file in os.listdir(path) if file.endswith(".xlsx") or file.endswith(".xls")]
        if files:
            return JSONResponse(
                content={
                    "message": "成功获取知识库Excel文件",
                    "files": files
                }
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "未找到Excel文件"}
            )
    except Exception as e:
        return e
    
@app.get("/get_knowledge_pdf")
async def get_knowledge_pdf():
    path = DATA_DIR_PDF
    try:
        files = [file for file in os.listdir(path) if file.endswith(".pdf")]
        if files:
            return JSONResponse(
                content={
                    "message": "成功获取知识库PDF文件",
                    "files": files
                }
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "未找到PDF文件"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Global variable to store MCP server thread
mcp_thread = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化资源"""
    # global mcp_thread
    # logger.info("正在启动应用和MCP服务...")
    
    # 初始化数据库
    init_database(EXCEL_DB_PATH)
    init_database(PDF_DB_PATH)
    
    # 在单独的线程中启动MCP服务
    # def start_mcp_server():
    #     try:
    #         logger.info("MCP服务正在启动...")
    #         run_mcp()
    #     except Exception as e:
    #         logger.error(f"MCP服务启动失败: {str(e)}")
    
    # mcp_thread = threading.Thread(target=start_mcp_server, daemon=True)
    # mcp_thread.start()
    # logger.info("MCP服务已在后台线程启动")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    # logger.info("正在关闭应用并清理资源...")
    # 关闭线程池
    thread_pool.shutdown(wait=True)
    # logger.info("应用已关闭，资源已清理完成")

if __name__ == "__main__":
    import uvicorn
    # logger.info("启动FastAPI应用和MCP服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
