import requests
import json
import time

base_url = "http://127.0.0.1:8000/query/excel"
data = {
    "question": "竣工验收确认各部门的主要职责是什么",
    "stream": True,
    "temperature": 0.5,
    "top_p": 0.95,
    "llm_top_k": 10,
    "conversation_id": "",
    "top_k": 10,
    "model": "Qwen3"
}

print("正在发送请求...")
response = requests.post(base_url, json=data, headers={"Content-Type": "application/json"}, stream=True)
print(f"响应状态码: {response.status_code}")
print("开始接收流式数据...")

# 处理流式响应
for chunk in response.iter_lines():
    if chunk:
        # 解码并打印数据
        decoded_chunk = chunk.decode('utf-8')
        try:
            # 处理SSE格式，移除data:前缀
            if decoded_chunk.startswith('data:'):
                json_str = decoded_chunk[5:].strip()
            else:
                json_str = decoded_chunk
            
            # 解析JSON数据
            data = json.loads(json_str)
            
            if 'answer' in data:
                print(data['answer'], end='', flush=True)
            elif 'error' in data:
                print(f"\n错误: {data['error']}")
            else:
                print(f"\n未知数据格式: {data}")
        except json.JSONDecodeError as e:
            print(f"\n解析JSON失败: {e}, 原始数据: {decoded_chunk}")
        except Exception as e:
            print(f"\n处理数据时出错: {e}")

print("\n流式响应接收完成")


# 导入必要的库
# from langchain.indexes import SQLRecordManager, index
# from langchain_core.documents import Document
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings

# docs_v1 = [
#     Document(page_content="LangChain是一个用于构建大语言模型应用的框架。", metadata={"source": "intro.txt"}),
#     Document(page_content="Chroma是一个轻量级的开源向量数据库。", metadata={"source": "vector_db.txt"}),
# ]

# # 1. 初始化嵌入模型和向量存储（使用持久化模式）
# embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest", base_url='http://10.81.38.110:11434')
# vectorstore = Chroma.from_documents(
#     documents=docs_v1,
#     collection_name="my_incremental_collection",
#     embedding=embedding,
#     persist_directory="./chroma_db"  # 数据将保存在本地此目录
# )

# # 2. 初始化记录管理器 (RecordManager)
# # namespace 建议格式：`向量库类型/集合名`，用于唯一标识管理记录[citation:2]
# namespace = "chroma/my_incremental_collection"
# record_manager = SQLRecordManager(
#     namespace,
#     db_url="sqlite:///record_manager.sqlite"  # 记录管理信息将存储在此SQLite文件
# )

# # 首次运行需要创建记录管理器的表结构[citation:2]
# record_manager.create_schema()

# # 3. 准备示例文档
# # 注意：metadata 中必须包含用于标识文档来源的字段（默认是 `source`）[citation:2][citation:5]

# # 4. 执行增量索引 (cleanup="incremental")
# print("=== 首次索引（新增两个文档）===")
# result = index(
#     docs_v1,
#     record_manager,
#     vectorstore,
#     cleanup="incremental",   # 启用增量模式
#     source_id_key="source",  # 指定元数据中标识文档来源的字段
# )
# print(f"索引结果: {result}")
# # 预期输出: {'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}

# # 5. 模拟文档更新后的增量索引
# # 假设 `intro.txt` 文件的内容发生了变化，我们创建一个新版本的文档
# docs_v1 = [
#     Document(page_content="LangChain是一个强大的框架，支持链式调用和工具集成。", metadata={"source": "intro.txt"}),  # 内容已更新
#     Document(page_content="Chroma是一个轻量级的开源向量数据库。", metadata={"source": "vector_db.txt"}),  # 内容未变
# ]

# print("\n=== 第二次索引（更新一个文档）===")
# result2 = index(
#     docs_v1,
#     record_manager,
#     vectorstore,
#     cleanup="incremental",
#     source_id_key="source",
# )
# print(f"索引结果: {result2}")


# docs_v1 = [
#     Document(page_content="LangChain是一个强大的框架，支持链式调用和工具集成。", metadata={"source": "intro.txt"}),  # 内容已更新 # 内容未变
# ]

# print("\n=== 第三次索引（删除一个文档）===")
# result3 = index(
#     docs_v1,
#     record_manager,
#     vectorstore,
#     cleanup="full",
#     source_id_key="source",
# )
# print(f"索引结果: {result3}")
# # 预期输出: {'num_added': 1, 'num_updated': 0, 'num_skipped': 1, 'num_deleted': 1}
# # 解释：新增了1个（更新的intro.txt），跳过了1个（未变的vector_db.txt），删除了1个（旧的intro.txt）