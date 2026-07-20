from pydantic import BaseModel
from typing import *
import sqlite3

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 8
    model: Optional[str] = "Qwen3-235B-A22B-Instruct-2507"
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.95
    llm_top_k: Optional[int] = 10
    stream: Optional[bool] = False

class QueryResponse(BaseModel):
    conversation_id: str
    answer: str

class DeleteConversationRequest(BaseModel):
    conversation_id: str

def save_conversation(db_path, conversation_id, user_question, assistant_answer):
    """保存对话历史到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 保存新的对话记录
    cursor.execute(
        'INSERT INTO conversation_history (conversation_id, user_question, assistant_answer) VALUES (?, ?, ?)',
        (conversation_id, user_question, assistant_answer)
    )
    
    # 检查并删除旧的对话记录，保留最多20条
    cursor.execute(
        'SELECT id FROM conversation_history WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT -1 OFFSET 20',
        (conversation_id,)
    )
    old_records = cursor.fetchall()
    for record_id, in old_records:
        cursor.execute('DELETE FROM conversation_history WHERE id = ?', (record_id,))
    
    conn.commit()
    conn.close()

def get_conversation_history(db_path, conversation_id):
    """从数据库获取对话历史"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_question, assistant_answer FROM conversation_history WHERE conversation_id = ? ORDER BY timestamp ASC',
        (conversation_id,)
    )
    history = cursor.fetchall()
    conn.close()
    return history

def delete_conversation_history(db_path, conversation_id):
    """删除指定对话ID的所有对话记录"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 删除指定对话ID的所有记录
    cursor.execute(
        'DELETE FROM conversation_history WHERE conversation_id = ?',
        (conversation_id,)
    )
    
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted_count