'''TextSplitter.py
Author: liyu
Date: 2025-09-11
update at : 2025-11-06 18:35
latest: 2025-11-10 19:17
UpdateLog:
    1. Create new WindowedTextSplitter to split the document into chunks by using sliding window strategy.
    2. Create EnhancedTextSplitter to split the document into chunks by using segment strategy.
    3. For excel and pdf document, we use different text splitter to split the document into chunks.
'''

import re
import json
from typing import *
from langchain_text_splitters import CharacterTextSplitter

try:
    from nltk.tokenize import sent_tokenize
    import nltk
except ImportError:
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs) -> None:
        super(ChineseTextSplitter, self).__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))') 
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list
    
class WindowedTextSplitter(CharacterTextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200) -> None:
        super(WindowedTextSplitter, self).__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        # 实现滑动窗口策略
        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start += self._chunk_size - self._chunk_overlap
            # 处理尾部剩余文本
            if end > len(text) and len(chunks) > 1:
                # 合并最后两个块避免过小
                last_chunk = chunks.pop()
                chunks[-1] += last_chunk
        return chunks
    

class BasicTextSplitter(CharacterTextSplitter):
    def __init__(self, chunk_size=300, chunk_overlap=100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            from nltk.tokenize import sent_tokenize
        except LookupError:
            import nltk
            nltk.download('punkt')
            from nltk.tokenize import sent_tokenize

    def split_text(self, text: str) -> List[str]:
        sentences = [s.strip() for p in text for s in sent_tokenize(p)]
        chunks = []
        i = 0
        while i < len(sentences):
            chunk = sentences[i]
            overlap = ''
            prev_len = 0
            prev = i - 1
            # 向前计算重叠部分
            while prev >= 0 and len(sentences[prev])+len(overlap) <= self.chunk_overlap:
                overlap = sentences[prev] + ' ' + overlap
                prev -= 1
            chunk = overlap+chunk
            next = i + 1
            # 向后计算当前chunk
            while next < len(sentences) and len(sentences[next])+len(chunk) <= self.chunk_size:
                chunk = chunk + ' ' + sentences[next]
                next += 1
            chunks.append(chunk)
            i = next
        return chunks

class EnhancedDocumentSplitter(CharacterTextSplitter):
    """
    增强型文档分割器，专为PDF和Excel文件优化，提高RAG检索精度。
    主要特点：
    1. 文档类型感知：自动适应PDF和Excel的不同格式特点
    2. 结构保留：保留文档的标题层次和段落结构
    3. 语义连贯性：基于句子和段落边界进行智能分割
    4. 动态重叠：根据内容重要性调整重叠策略
    5. 多语言优化：更好地处理中英文混合文本
    """
    
    def __init__(self, 
                 chunk_size=500, 
                 chunk_overlap=100,
                 min_chunk_size=100,
                 document_type='pdf',  # 'pdf' 或 'excel'
                 preserve_structure=True,
                 semantic_split=True,
                 **kwargs):
        """
        初始化增强型文档分割器
        
        Args:
            chunk_size: 块的目标大小
            chunk_overlap: 块之间的重叠大小
            min_chunk_size: 最小块大小
            document_type: 文档类型，'pdf' 或 'excel'
            preserve_structure: 是否保留文档结构
            semantic_split: 是否启用语义分割
        """
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.document_type = document_type.lower()
        self.preserve_structure = preserve_structure
        self.semantic_split = semantic_split
        
        # 初始化标题检测模式
        self._init_patterns()
        
    def _init_patterns(self):
        """初始化正则表达式模式"""
        # 标题模式
        self.heading_patterns = [
            # Markdown风格标题
            (re.compile(r'^#{1,6}\s+'), 1),
            # 多级编号标题 (1. 2.1 3.1.1)
            (re.compile(r'^((\d+\.)+\d+|\d+)\s+'), 2),
            # 中文编号标题 (一、 二、 1、 2、)
            (re.compile(r'^[一二三四五六七八九十百千]+、\s+|^\d+\s+'), 2),
            # 方括号标题
            (re.compile(r'^[【\[](.*?)[】\]]\s+'), 3),
            # 加粗标题
            (re.compile(r'^\*{1,3}(.*?)\*{1,3}\s+'), 3),
            # 大写字母编号
            (re.compile(r'^[A-Z]\.\s+'), 2)
        ]
        
        # 段落分隔符
        self.paragraph_separator = re.compile(r'\n\s*\n')
        
        # 句子结束符（中英文）
        self.sentence_enders = re.compile(r'[。！？.!?]["”’）)》＞\]\}]*\s*')
        
        # Excel特有模式
        if self.document_type == 'excel':
            # 表格标题模式
            self.table_header_pattern = re.compile(r'^[A-Z]+\d*\s*[:：]')
            # 数据行模式
            self.data_row_pattern = re.compile(r'^\d+[\t\s]+.*')
    
    def _is_heading(self, text: str) -> tuple:
        """检测文本是否为标题，并返回级别
        
        Args:
            text: 待检测文本
            
        Returns:
            (是否为标题, 标题级别)
        """
        if not text or len(text.strip()) < 3:
            return False, 0
            
        stripped = text.strip()
        for pattern, level in self.heading_patterns:
            if pattern.match(stripped):
                # 根据匹配的标题样式确定级别
                if level == 1:  # Markdown
                    # 根据#的数量确定级别
                    heading_level = len(pattern.match(stripped).group(0)) - 1
                    return True, heading_level
                elif level == 2:  # 编号
                    # 计算点的数量+1作为级别
                    dot_count = stripped.count('.')
                    return True, dot_count + 1
                return True, level
        
        # 特殊判断：如果文本全大写或以冒号结尾且较短，可能是标题
        if len(stripped) < 50 and (stripped.isupper() or stripped.endswith((':', '：'))):
            return True, 4
            
        return False, 0
    
    def _segment_sentences(self, text: str) -> List[str]:
        """分割文本为句子列表
        
        Args:
            text: 待分割文本
            
        Returns:
            句子列表
        """
        # 先尝试使用正则表达式分割
        sentences = []
        pos = 0
        
        for match in self.sentence_enders.finditer(text):
            end_pos = match.end()
            sentence = text[pos:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            pos = end_pos
        
        # 处理最后一个句子
        if pos < len(text):
            last_sentence = text[pos:].strip()
            if last_sentence:
                sentences.append(last_sentence)
        
        # 如果正则分割效果不好（句子太少或太长），使用nltk作为后备
        if len(sentences) <= 1 and len(text) > 200:
            try:
                sentences = sent_tokenize(text)
            except:
                # 如果nltk也失败，进行简单的字符分割
                pass
        
        return sentences
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本，根据文档类型进行特定处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 基础清理
        text = text.strip()
        
        if self.document_type == 'pdf':
            # PDF特定处理
            # 移除多余空行
            text = re.sub(r'\n{3,}', '\n\n', text)
            # 修复断行（行尾没有标点的情况下连接到下一行）
            lines = text.split('\n')
            processed_lines = []
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line:
                    # 检查是否需要连接到下一行
                    if (i < len(lines) - 1 and 
                        not self.sentence_enders.search(line) and 
                        not self._is_heading(line)[0]):
                        # 行尾没有结束符，可能是断行
                        processed_lines.append(stripped_line + ' ')
                    else:
                        processed_lines.append(stripped_line)
            text = '\n'.join(processed_lines)
            
        elif self.document_type == 'excel':
            # Excel特定处理
            # 处理表格结构，保留列名信息
            lines = text.split('\n')
            processed_lines = []
            headers = []
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                    
                # 检测表格标题
                if self.table_header_pattern.match(stripped):
                    headers.append(stripped.split(':', 1)[0].strip())
                    processed_lines.append(f"[表格标题: {stripped}]")
                # 检测数据行
                elif self.data_row_pattern.match(stripped) and headers:
                    # 在数据行前添加相关表头信息，增强上下文
                    context_line = f"[数据行({', '.join(headers)}): {stripped}]"
                    processed_lines.append(context_line)
                else:
                    processed_lines.append(stripped)
                    
            text = '\n'.join(processed_lines)
        
        return text
    
    def split_text(self, text: str) -> List[str]:
        """分割文本为块
        
        Args:
            text: 待分割文本
            
        Returns:
            文本块列表
        """
        # 预处理文本
        text = self._preprocess_text(text)
        
        if not self.preserve_structure or not self.semantic_split:
            # 基本分割模式
            return self._basic_split(text)
        
        # 结构感知分割
        return self._structure_aware_split(text)
    
    def _basic_split(self, text: str) -> List[str]:
        """基本文本分割
        
        Args:
            text: 待分割文本
            
        Returns:
            文本块列表
        """
        # 先分割段落
        paragraphs = self.paragraph_separator.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 如果添加当前段落会超过块大小，考虑分割
            if len(current_chunk) + len(para) + 1 > self.chunk_size:
                # 如果当前块不为空，先保存
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果段落本身就很大，需要进一步分割
                if len(para) > self.chunk_size:
                    sentences = self._segment_sentences(para)
                    # 将大段落按句子分割
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 1 > self.chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                                # 考虑重叠：将最后几个句子作为下一个块的开头
                                overlap_size = min(self.chunk_overlap, len(temp_chunk))
                                # 从后往前找句子边界
                                overlap_end = len(temp_chunk)
                                overlap_start = max(0, overlap_end - overlap_size)
                                # 尽量在句子边界分割重叠部分
                                overlap_text = temp_chunk[overlap_start:]
                                temp_chunk = overlap_text
                        
                        if temp_chunk:
                            temp_chunk += " " + sentence
                        else:
                            temp_chunk = sentence
                    
                    if temp_chunk:  # 保存最后一个部分
                        chunks.append(temp_chunk)
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 后处理：合并过小的块
        return self._postprocess_chunks(chunks)
    
    def _structure_aware_split(self, text: str) -> List[str]:
        """结构感知的文本分割
        
        Args:
            text: 待分割文本
            
        Returns:
            文本块列表
        """
        # 按行分割
        lines = text.split('\n')
        sections = []
        current_section = {'content': [], 'level': 0, 'is_heading': False}
        
        # 第一阶段：识别结构，按标题分割
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            is_heading, level = self._is_heading(stripped_line)
            
            if is_heading:
                # 如果当前章节内容不为空，先保存当前章节
                if current_section['content']:
                    sections.append(current_section)
                
                # 开始新章节
                current_section = {
                    'content': [stripped_line],
                    'level': level,
                    'is_heading': True
                }
            else:
                # 添加内容到当前章节
                current_section['content'].append(line)
        
        # 处理最后一个章节
        if current_section['content']:
            sections.append(current_section)
        
        # 第二阶段：对每个章节进行语义分割
        final_chunks = []
        
        for section in sections:
            section_text = '\n'.join(section['content'])
            
            # 如果章节很小，可以直接作为一个块
            if len(section_text) < self.chunk_size * 0.8:  # 80%的块大小作为阈值
                # 检查是否需要与前一个块合并（如果是小节）
                if section['level'] > 2 and final_chunks:
                    # 尝试合并到前一个块
                    if len(final_chunks[-1]) + len(section_text) < self.chunk_size * 1.2:
                        final_chunks[-1] += '\n' + section_text
                        continue
                
                final_chunks.append(section_text)
            else:
                # 大章节需要进一步分割
                # 先分割段落
                paragraphs = self.paragraph_separator.split(section_text)
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                
                temp_chunk = ""
                for para in paragraphs:
                    # 确保章节标题始终在块的开头
                    if section['is_heading'] and temp_chunk == "":
                        temp_chunk = para  # 章节标题作为第一个段落
                        continue
                    
                    # 检查是否需要分割
                    if len(temp_chunk) + len(para) + 1 > self.chunk_size:
                        if temp_chunk:
                            final_chunks.append(temp_chunk)
                            # 智能重叠：保留前一个块的最后一部分
                            overlap_size = min(self.chunk_overlap, len(temp_chunk) // 2)
                            overlap_start = max(0, len(temp_chunk) - overlap_size)
                            # 尝试在句子边界分割重叠部分
                            sentences = self._segment_sentences(temp_chunk[overlap_start:])
                            # 保留最后2-3个句子作为重叠
                            overlap_sentences = sentences[-min(3, len(sentences)):]
                            temp_chunk = " ".join(overlap_sentences)
                    
                    # 添加当前段落
                    if temp_chunk:
                        temp_chunk += "\n" + para
                    else:
                        temp_chunk = para
                
                # 保存最后一个块
                if temp_chunk:
                    final_chunks.append(temp_chunk)
        
        # 后处理
        return self._postprocess_chunks(final_chunks)
    
    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        """后处理文本块
        
        Args:
            chunks: 原始文本块列表
            
        Returns:
            处理后的文本块列表
        """
        processed = []
        
        for i, chunk in enumerate(chunks):
            # 清理空白
            chunk = chunk.strip()
            
            # 处理过小的块
            if len(chunk) < self.min_chunk_size:
                # 尝试与前一个块合并
                if processed:
                    processed[-1] += "\n" + chunk
                # 如果是第一个块，尝试与下一个块合并
                elif i < len(chunks) - 1:
                    continue  # 跳过，在下一轮合并
                else:
                    # 无法合并，保留原样
                    processed.append(chunk)
            else:
                processed.append(chunk)
        
        # 最后检查是否有未合并的小块
        if len(chunks) > 1 and len(chunks[-1]) < self.min_chunk_size and len(processed) > 0:
            processed[-1] += "\n" + chunks[-1]
        
        return processed

class DocumentTypeAdapter:
    """
    文档类型适配器，根据输入文档类型自动选择最佳分割策略
    """
    
    @staticmethod
    def get_splitter(file_path: str = None, document_type: str = None, **kwargs) -> EnhancedDocumentSplitter:
        """
        根据文件路径或文档类型获取合适的分割器
        
        Args:
            file_path: 文件路径
            document_type: 文档类型
            **kwargs: 传递给分割器的参数
            
        Returns:
            适当配置的分割器实例
        """
        # 确定文档类型
        doc_type = "pdf"  # 默认
        
        if document_type:
            doc_type = document_type.lower()
        elif file_path:
            # 从文件扩展名推断
            if file_path.lower().endswith(('.xls', '.xlsx')):
                doc_type = "excel"
            elif file_path.lower().endswith('.pdf'):
                doc_type = "pdf"
        
        # 根据文档类型调整参数
        if doc_type == "excel":
            # Excel文档通常更结构化，块大小可以小一些
            kwargs.setdefault('chunk_size', 400)
            kwargs.setdefault('chunk_overlap', 80)
            kwargs.setdefault('preserve_structure', True)
        else:  # pdf或其他
            kwargs.setdefault('chunk_size', 500)
            kwargs.setdefault('chunk_overlap', 100)
        
        return EnhancedDocumentSplitter(document_type=doc_type, **kwargs)
