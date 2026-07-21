# Document QA FastAPI Application

基于FastAPI的文档问答系统，支持Excel、PDF文档以及飞书在线文档的智能问答，具备自动外部检索能力和文件管理功能。
项目仅供参考和学习

## 功能特性

- 支持Excel文档问答 (`/query/excel`)
- 支持PDF文档问答 (`/query/pdf`)
- 支持飞书在线文档问答 (`/query/excel` - 使用LarkSuiteOnlineRAG)
- 文档文件上传：支持批量上传Excel和PDF文件 (`/upload/excel`, `/upload/pdf`)
- 文档文件删除：支持删除知识库中的文件 (`/delete_file`)
- 向量存储更新：支持手动更新文档向量存储 (`/update/excel`, `/update/pdf`)
- 知识库文件列表获取：查看当前知识库中的文件 (`/get_knowledge_excel`, `/get_knowledge_pdf`)
- 对话历史管理，支持多用户并发请求
- 支持流式响应和非流式响应
- 上下文感知的问答处理
- 自动外部知识检索：当大模型无法回答问题时，自动切换到外部搜索引擎获取答案
- 对话历史保存：每个对话ID最多保存20条历史记录
- 健康检查：实时监控服务状态 (`/health`)
- 支持混合检索策略（向量检索 + BM25）
- 多模态RAG（正在完善与开发）

## API接口

### 1. Excel文档问答

**POST** `/query/excel`

请求体：
```json
{
  "question": "用户的提问",
  "conversation_id": "用户的对话ID，用于区分不同的对话会话，当输入为空时，默认创建一个新的对话会话",
  "stream": "是否流式返回回答，默认值为False",
  "top_k": "检索文档的Top K个，默认Excel8个",
  "model": "使用的模型，默认值为Qwen3-235B-A22B-Instruct-2507",
  "temperature": "生成回答的温度参数，默认值为0.5",
  "top_p": "生成回答的Top P参数，默认值为0.95",
  "llm_top_k": "生成回答的Top K个，默认值为10"
}
```

请求示例：
```bash
curl -X POST "http://localhost:8000/query/excel" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "<USER QUESTION>",
    "conversation_id": "conv_123456",
    "stream": false,
    "top_k": 10,
    "model": "Qwen3-235B-A22B-Instruct-2507",
    "temperature": 0.5,
    "top_p": 0.95,
    "llm_top_k": 10
  }'
```

响应：
```json
{
  "conversation_id": "用户的对话ID，用于区分不同的对话会话",
  "answer": "模型的回答"
}
```

响应示例：
```json
{
  "conversation_id": "conv_123456",
  "answer": "<MODEL OUTPUT>"
}
```

对于流式响应，返回格式为Server-Sent Events (SSE)，每条消息格式为：
```
data: {"conversation_id": "对话ID", "answer": "回答内容片段"}
```

流式响应示例：
```bash
curl -X POST "http://localhost:8000/query/excel" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "<USER QUESTION>",
    "stream": true
  }'
```

### 2. PDF文档问答

**POST** `/query/pdf`

请求体和响应格式与Excel接口相同，默认Top K为30个。

请求示例：
```bash
curl -X POST "http://localhost:8000/query/pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "question": <USER QUESTION>,
    "conversation_id": "conv_789012",
    "stream": false,
    "top_k": 30
  }'
```

响应示例：
```json
{
  "conversation_id": "conv_789012",
  "answer": "<MODEL OUTPUT>"
}
```

### 3. 上传Excel文件

**POST** `/upload/excel`

请求示例：
```bash
curl -X POST "http://localhost:8000/upload/excel" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@file1.xlsx" \
  -F "files=@file2.xlsx"
```

响应：
```json
{
  "filenames": ["file1.xlsx", "file2.xlsx"]
}
```

### 4. 上传PDF文件

**POST** `/upload/pdf`

请求示例：
```bash
curl -X POST "http://localhost:8000/upload/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@file1.pdf" \
  -F "files=@file2.pdf"
```

响应：
```json
{
  "filenames": ["file1.pdf", "file2.pdf"]
}
```

### 5. 更新Excel向量存储

**POST** `/update/excel`

请求示例：
```bash
curl -X POST "http://localhost:8000/update/excel"
```

响应：
```json
{
  "message": "Excel RAG实例更新完成"
}
```

### 6. 更新PDF向量存储

**POST** `/update/pdf`

请求示例：
```bash
curl -X POST "http://localhost:8000/update/pdf"
```

响应：
```json
{
  "message": "PDF RAG实例更新完成"
}
```

### 7. 删除对话历史

**DELETE** `/delete/conversation`

请求体：
```json
{
  "conversation_id": "用户的对话ID"
}
```

请求示例：
```bash
curl -X DELETE "http://localhost:8000/delete/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123456"
  }'
```

响应：
```json
{
  "message": "对话记录已成功删除",
  "conversation_id": "用户的对话ID",
  "deleted_count": 删除的记录总数
}
```

### 8. 删除知识库文件

**DELETE** `/delete_file`

请求体：
```json
[
  "file1.xlsx",
  "file1.pdf"
]
```

请求示例：
```bash
curl -X DELETE "http://localhost:8000/delete_file" \
  -H "Content-Type: application/json" \
  -d '["file1.xlsx", "file2.pdf"]'
```

响应：
```json
{
  "message": "文件已成功删除",
  "file": "文件名"
}
```

### 9. 获取Excel知识库文件列表

**GET** `/get_knowledge_excel`

请求示例：
```bash
curl -X GET "http://localhost:8000/get_knowledge_excel"
```

响应：
```json
{
  "message": "成功获取知识库Excel文件",
  "files": ["file1.xlsx", "file2.xlsx"]
}
```

### 10. 获取PDF知识库文件列表

**GET** `/get_knowledge_pdf`

请求示例：
```bash
curl -X GET "http://localhost:8000/get_knowledge_pdf"
```

响应：
```json
{
  "message": "成功获取知识库PDF文件",
  "files": ["file1.pdf", "file2.pdf"]
}
```

### 11. 健康检查

**GET** `/health`

请求示例：
```bash
curl -X GET "http://localhost:8000/health"
```

响应：
```json
{
  "status": "healthy",
  "service": "DocumentQA API"
}
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

```bash
python fastapi_application.py
```

或者使用uvicorn：

```bash
uvicorn fastapi_application:app --host 0.0.0.0 --port 8000
```

## 并发支持

应用支持多用户并发请求，每个接口的并发请求数不超过100个。使用线程池处理请求，确保检索和生成的速度和质量不受影响。

## 对话历史管理

- 对话历史保存在SQLite数据库中，使用`conversation_id`进行区分
- 当使用已生成的`conversation_id`再次请求时，不会生成新的`conversation_id`
- 每个对话ID最多保存20条最近的历史记录
- 可通过`DELETE /delete/conversation`接口删除指定会话的全部对话历史

## 环境变量

确保设置以下环境变量：
- `OpenAI_API_Key`: OpenAI API密钥，用于调用LLM服务,如果是本地部署的大模型可以忽略
- `SERPAPI_API_KEY`: SerpAPI密钥，用于外部知识检索

## 特殊功能说明

### 外部知识检索
当问题中包含"外部知识"关键词时，系统会自动切换到外部搜索引擎获取答案。

### 知识库与外部知识结合
当问题中同时包含"外部知识"和"知识库"关键词时，系统会同时使用知识库和外部搜索来提供更全面的答案。

## 系统架构

- 后端：FastAPI
- 数据库：SQLite（用于存储对话历史）
- 向量存储：Chroma（用于文档检索）
- 搜索引擎：DuckDuckGo（用于外部知识检索，通过MCP工具）
- 模型：
  - 语言模型：Qwen3-235B-A22B-Instruct-2507（OpenAI兼容接口）
  - 嵌入模型：quentinz/bge-large-zh-v1.5（Ollama）
- 检索算法：向量检索 + BM25混合检索

## 配置说明

配置文件位于`models/config.py`，可调整以下参数：
- `DATA_DIR_PDF`: PDF文档目录
- `VS_PATH_PDF`: PDF向量存储路径
- `DATA_DIR_EXCEL`: Excel文档目录
- `VS_PATH_EXCEL`: Excel向量存储路径
- `VS_PATH_LARKSUITE`: 飞书文档向量存储路径
- `CHUNK_SIZE`: 文档分块大小，默认100000
- `CHUNK_OVERLAP`: 文档分块重叠大小，默认4000
- `VECTOR_SEARCH_TOP_K_EXCEL`: Excel文档检索Top K，默认8
- `VECTOR_SEARCH_TOP_K_PDF`: PDF文档检索Top K，默认30
- `USE_MMR`: 是否使用最大边际相关性，默认True
- `MMR_LAMBDA`: MMR参数，默认0.5
- `VECTOR_WEIGHT`: 向量检索权重，默认0.7
- `BM25_WEIGHT`: BM25检索权重，默认0.3
