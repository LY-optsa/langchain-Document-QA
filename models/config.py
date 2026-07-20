'''
Configuration file for the project.
Author: liyu
Date: 2025-09-11
Update:2025-11-10 19:20
UpdateLog:
1. Modify the value of CHUNK_SIZE and CHUNK_OVERLAP for the project.
2. Rewrite Chat Template for the project, update the template to support multi-turn conversation.
'''

import os
DATA_DIR_PDF = os.path.join(os.path.dirname(__file__), "../pdf")
VS_PATH_PDF = os.path.join(os.path.dirname(__file__), "../vector_store_pdf/Chroma")
DATA_DIR_EXCEL = os.path.join(os.path.dirname(__file__), "../excel")
VS_PATH_EXCEL = os.path.join(os.path.dirname(__file__), "../vector_store_excel/Chroma")
VS_PATH_LARKSUITE = os.path.join(os.path.dirname(__file__), "../vector_store_larksuite/Chroma")
CHUNK_SIZE = 100000
CHUNK_OVERLAP = 4000
VECTOR_SEARCH_TOP_K_EXCEL = 8
VECTOR_SEARCH_TOP_K_PDF = 30
os.environ["SERPAPI_API_KEY"] = "Your SerpAPI Key"
USE_MMR = True
MMR_LAMBDA = 0.5
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

TEMPLATE = """
###应知应会及企业政策文档问答助手###
你需要根据提供的对话历史和上下文来回答用户的最新提问
注意：
1 请确保你的回答与对话历史保持连贯性，基于之前的问答内容进行回复
2 若上下文内容不足以回答问题，请回复：根据您提供的上下文信息，我暂时无法回答你的问题，我将使用联网搜索工具来获取更多信息。
3 若上下文信息为空，请回复：您似乎没有提供上下文信息，作为文档问答助手，我暂时无法回答你的问题。我将使用联网搜索工具来获取更多信息。
4 若上下文信息可以完整回答用户的问题，回答过程中，你需要给出相应文档的出处，即在哪个文档的哪个位置，如果是excel文档，你需要给出excel文件的名称和sheet名称。
5 若对话历史不为空，你需要同时依据上下文和对话历史回答问题
对话历史：{chat_history}
最新提问：{question}
上下文：{context}
回答：
"""