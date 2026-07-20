from typing import List, Dict, Any, Tuple, Union
import os
import re
import json
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from models.TextSplitter import EnhancedDocumentSplitter, DocumentTypeAdapter
from models.config import *
from bs4 import BeautifulSoup

class MultimodalDocumentProcessor:
    """
    多模态文档处理器，实现多模态RAG的完整流程
    包括：文档分割、内容提取、HTML转换、语义分块、向量化存储
    """
    
    def __init__(self):
        # 初始化CLIP模型用于图像向量化
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./cache")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./cache")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        
        # 创建临时目录用于存储中间结果
        self.temp_dir = "temp_mm_rag"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def structure_preserving_document_split(self, file_path: str) -> List[Dict[str, Any]]:
        """
        结构保留的文档分割：将PDF文档分解为可管理的片段，保持逻辑结构
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            结构化文档片段列表，每个片段包含类型、内容、位置等信息
        """
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("目前仅支持PDF文件")
        
        doc = fitz.open(file_path)
        structured_segments = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_dict = {
                "page_number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "segments": []
            }
            
            # 提取文本块
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                if block[6] == 0:  # 文本块
                    text_segment = {
                        "type": "text",
                        "content": block[4],
                        "bbox": {
                            "x0": block[0],
                            "y0": block[1],
                            "x1": block[2],
                            "y1": block[3]
                        },
                        "order": len(page_dict["segments"])
                    }
                    page_dict["segments"].append(text_segment)
            
            # 提取图片
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 保存图片到临时目录
                image_path = os.path.join(self.temp_dir, f"page_{page_num+1}_img_{img_index}.png")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # 获取图片在页面中的位置
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    img_bbox = img_rects[0]
                    image_segment = {
                        "type": "image",
                        "content": image_path,
                        "bbox": {
                            "x0": img_bbox.x0,
                            "y0": img_bbox.y0,
                            "x1": img_bbox.x1,
                            "y1": img_bbox.y1
                        },
                        "order": len(page_dict["segments"])
                    }
                    page_dict["segments"].append(image_segment)
            
            # 提取表格
            tables = page.find_tables()
            for table_index, table in enumerate(tables):
                table_data = table.extract()
                if table_data:
                    table_df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    table_html = table_df.to_html(index=False)
                    
                    table_segment = {
                        "type": "table",
                        "content": table_html,
                        "raw_data": table_data,
                        "bbox": {
                            "x0": table.bbox[0],
                            "y0": table.bbox[1],
                            "x1": table.bbox[2],
                            "y1": table.bbox[3]
                        },
                        "order": len(page_dict["segments"])
                    }
                    page_dict["segments"].append(table_segment)
            
            # 提取公式（基于文本特征识别）
            # 查找可能是公式的文本块（包含数学符号、特殊格式等）
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                if block[6] == 0:  # 文本块
                    text = block[4]
                    # 简单的公式识别：包含数学符号或希腊字母
                    if re.search(r'[=+\-*/^()\[\]{}<>≈≠≤≥±∑∫∂∇Δπ\αβγδεζηθικλμνξοπρστυφχψω]', text):
                        # 检查是否已经作为普通文本处理过
                        already_processed = False
                        for seg in page_dict["segments"]:
                            if seg["type"] == "text" and seg["content"] == text:
                                already_processed = True
                                break
                        
                        if not already_processed:
                            formula_segment = {
                                "type": "formula",
                                "content": text,
                                "bbox": {
                                    "x0": block[0],
                                    "y0": block[1],
                                    "x1": block[2],
                                    "y1": block[3]
                                },
                                "order": len(page_dict["segments"])
                            }
                            page_dict["segments"].append(formula_segment)
            
            # 按位置排序所有片段
            page_dict["segments"].sort(key=lambda x: (x["bbox"]["y0"], x["bbox"]["x0"]))
            
            # 为非文本块添加标题和说明
            segments = page_dict["segments"]
            for i, segment in enumerate(segments):
                if segment["type"] in ["table", "image", "formula"]:
                    # 初始化标题和说明
                    title = ""
                    caption = ""
                    
                    # 查找上方的标题（通常在上方200像素内，水平重叠度高）
                    for j in range(i-1, max(-1, i-10), -1):  # 检查前10个片段
                        prev_seg = segments[j]
                        if prev_seg["type"] == "text":
                            # 计算垂直距离和水平重叠度
                            vertical_dist = segment["bbox"]["y0"] - prev_seg["bbox"]["y1"]
                            if vertical_dist < 0:  # 前一个片段在当前片段下方，跳过
                                continue
                            if vertical_dist > 200:  # 距离太远，不是标题
                                break
                            
                            # 计算水平重叠度
                            overlap_left = max(segment["bbox"]["x0"], prev_seg["bbox"]["x0"])
                            overlap_right = min(segment["bbox"]["x1"], prev_seg["bbox"]["x1"])
                            overlap_width = max(0, overlap_right - overlap_left)
                            avg_width = (segment["bbox"]["x1"] - segment["bbox"]["x0"] + prev_seg["bbox"]["x1"] - prev_seg["bbox"]["x0"]) / 2
                            overlap_ratio = overlap_width / avg_width if avg_width > 0 else 0
                            
                            if overlap_ratio > 0.5:  # 重叠度超过50%，可能是标题
                                title = prev_seg["content"].strip()
                                break
                    
                    # 查找下方的说明（通常在下方100像素内，水平重叠度高）
                    for j in range(i+1, min(len(segments), i+5)):  # 检查后5个片段
                        next_seg = segments[j]
                        if next_seg["type"] == "text":
                            # 计算垂直距离和水平重叠度
                            vertical_dist = next_seg["bbox"]["y0"] - segment["bbox"]["y1"]
                            if vertical_dist < 0:  # 后一个片段在当前片段上方，跳过
                                continue
                            if vertical_dist > 100:  # 距离太远，不是说明
                                break
                            
                            # 计算水平重叠度
                            overlap_left = max(segment["bbox"]["x0"], next_seg["bbox"]["x0"])
                            overlap_right = min(segment["bbox"]["x1"], next_seg["bbox"]["x1"])
                            overlap_width = max(0, overlap_right - overlap_left)
                            avg_width = (segment["bbox"]["x1"] - segment["bbox"]["x0"] + next_seg["bbox"]["x1"] - next_seg["bbox"]["x0"]) / 2
                            overlap_ratio = overlap_width / avg_width if avg_width > 0 else 0
                            
                            if overlap_ratio > 0.3:  # 重叠度超过30%，可能是说明
                                caption = next_seg["content"].strip()
                                break
                    
                    # 添加标题和说明到片段
                    segment["title"] = title
                    segment["caption"] = caption
            
            structured_segments.append(page_dict)
        
        doc.close()
        return structured_segments
    
    def modality_specific_content_extraction(self, structured_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        模态特定内容提取：针对不同模态内容使用专用工具处理
        
        Args:
            structured_segments: 结构化文档片段列表
            
        Returns:
            处理后的多模态内容列表
        """
        processed_segments = []
        
        for page in structured_segments:
            page_processed = page.copy()
            page_processed["segments"] = []
            
            for segment in page["segments"]:
                processed_segment = segment.copy()
                
                if segment["type"] == "text":
                    # 文本处理：清理、格式化
                    text = segment["content"]
                    # 移除多余空格和换行
                    text = re.sub(r'\s+', ' ', text).strip()
                    processed_segment["content"] = text
                
                elif segment["type"] == "image":
                    # 图像处理：基本元数据提取
                    try:
                        with Image.open(segment["content"]) as img:
                            processed_segment["metadata"] = {
                                "width": img.width,
                                "height": img.height,
                                "format": img.format
                            }
                    except Exception as e:
                        processed_segment["metadata"] = {"error": str(e)}
                
                elif segment["type"] == "table":
                    # 表格处理：增强结构化信息
                    # 确保表格有正确的列名
                    table_data = segment["raw_data"]
                    if len(table_data) > 0:
                        # 检查第一行是否为有效的列名
                        first_row = table_data[0]
                        if all(isinstance(cell, str) and cell.strip() != "" for cell in first_row):
                            # 第一行是列名
                            processed_segment["has_header"] = True
                        else:
                            # 第一行不是列名，生成默认列名
                            processed_segment["has_header"] = False
                            num_cols = len(first_row)
                            table_df = pd.DataFrame(table_data, columns=[f"列{i+1}" for i in range(num_cols)])
                            processed_segment["content"] = table_df.to_html(index=False)
                
                elif segment["type"] == "formula":
                    # 公式处理：清理、格式化
                    formula = segment["content"]
                    # 移除多余空格和换行
                    formula = re.sub(r'\s+', ' ', formula).strip()
                    processed_segment["content"] = formula
                    # 保留标题和说明
                    processed_segment["title"] = segment.get("title", "")
                    processed_segment["caption"] = segment.get("caption", "")
                
                # 为非文本块保留标题和说明
                if segment["type"] in ["table", "image"]:
                    processed_segment["title"] = segment.get("title", "")
                    processed_segment["caption"] = segment.get("caption", "")
                
                page_processed["segments"].append(processed_segment)
            
            processed_segments.append(page_processed)
        
        return processed_segments
    
    def relationship_preserving_html_conversion(self, processed_segments: List[Dict[str, Any]]) -> str:
        """
        关系保留的HTML转换：将提取的多模态内容转换为结构化HTML，保留元素间关联
        
        Args:
            processed_segments: 处理后的多模态内容列表
            
        Returns:
            结构化HTML字符串
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='zh-CN'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>多模态文档</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; line-height: 1.6; }",
            ".page { margin: 20px 0; padding: 20px; border: 1px solid #ddd; page-break-after: always; }",
            ".text-block { margin: 10px 0; padding: 5px; }",
            ".image-block { margin: 15px 0; text-align: center; }",
            ".image-block img { max-width: 100%; height: auto; border: 1px solid #eee; }",
            ".table-block { margin: 15px 0; overflow-x: auto; }",
            ".table-block th, .table-block td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            ".table-block th { background-color: #f2f2f2; }",
            ".formula-block { margin: 15px 0; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #ccc; font-family: 'Courier New', monospace; }",
            ".segment { margin: 5px 0; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        for page_idx, page in enumerate(processed_segments):
            html_parts.append(f"<div class='page' data-page='{page['page_number']}'>")
            html_parts.append(f"<h2>第 {page['page_number']} 页</h2>")
            
            for seg_idx, segment in enumerate(page['segments']):
                segment_id = f"seg_{page['page_number']}_{seg_idx}"
                
                if segment["type"] == "text":
                    html_parts.append(f"<div class='segment text-block' id='{segment_id}' data-type='text'>")
                    html_parts.append(f"<p>{segment['content']}</p>")
                    html_parts.append("</div>")
                
                elif segment["type"] == "image":
                    rel_path = os.path.basename(segment["content"])
                    html_parts.append(f"<div class='segment image-block' id='{segment_id}' data-type='image'>")
                    if "title" in segment and segment["title"]:
                        html_parts.append(f"<h3 class='segment-title'>{segment['title']}</h3>")
                    html_parts.append(f"<img src='{rel_path}' alt='图片 {seg_idx+1}' />")
                    if "metadata" in segment:
                        html_parts.append(f"<p>图片信息: {segment['metadata']['width']}x{segment['metadata']['height']}px</p>")
                    if "caption" in segment and segment["caption"]:
                        html_parts.append(f"<p class='segment-caption'>{segment['caption']}</p>")
                    html_parts.append("</div>")
                
                elif segment["type"] == "table":
                    html_parts.append(f"<div class='segment table-block' id='{segment_id}' data-type='table'>")
                    if "title" in segment and segment["title"]:
                        html_parts.append(f"<h3 class='segment-title'>{segment['title']}</h3>")
                    html_parts.append(segment['content'])
                    if "caption" in segment and segment["caption"]:
                        html_parts.append(f"<p class='segment-caption'>{segment['caption']}</p>")
                    html_parts.append("</div>")
                
                elif segment["type"] == "formula":
                    html_parts.append(f"<div class='segment formula-block' id='{segment_id}' data-type='formula'>")
                    if "title" in segment and segment["title"]:
                        html_parts.append(f"<h3 class='segment-title'>{segment['title']}</h3>")
                    html_parts.append(f"<p>{segment['content']}</p>")
                    if "caption" in segment and segment["caption"]:
                        html_parts.append(f"<p class='segment-caption'>{segment['caption']}</p>")
                    html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        html_parts.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def relationship_preserving_semantic_chunking(self, html_content: str) -> List[Dict[str, Any]]:
        """
        关系保留的语义分块：将HTML内容划分为语义完整的片段，维护元素间关联
        
        Args:
            html_content: 结构化HTML字符串
            
        Returns:
            语义块列表
        """
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        pages = soup.find_all('div', class_='page')
        
        semantic_chunks = []
        chunk_id = 0
        
        for page in pages:
            page_num = page.get('data-page')
            segments = page.find_all('div', class_='segment')
            
            current_chunk = {
                "id": f"chunk_{chunk_id}",
                "page": page_num,
                "content": [],
                "types": set(),
                "relations": []
            }
            
            for i, segment in enumerate(segments):
                seg_type = segment.get('data-type')
                seg_content = str(segment)
                
                # 检查当前块是否可以容纳新片段
                # 简单策略：块大小不超过阈值，且保持语义连贯性
                current_size = sum(len(str(item['content'])) for item in current_chunk['content'])
                new_segment_size = len(seg_content)
                
                # 如果当前块为空，或者添加后不超过阈值，且类型连贯，则添加到当前块
                if not current_chunk['content'] or (
                    current_size + new_segment_size < 1000 and
                    (seg_type in current_chunk['types'] or len(current_chunk['content']) < 3)
                ):
                    # 添加到当前块
                    current_chunk['content'].append({
                        "type": seg_type,
                        "content": seg_content
                    })
                    current_chunk['types'].add(seg_type)
                    
                    # 添加关系：与前一个片段的顺序关系
                    if len(current_chunk['content']) > 1:
                        current_chunk['relations'].append({
                            "from": len(current_chunk['content']) - 2,
                            "to": len(current_chunk['content']) - 1,
                            "type": "sequential"
                        })
                else:
                    # 保存当前块
                    if current_chunk['content']:
                        semantic_chunks.append(current_chunk)
                        chunk_id += 1
                    
                    # 开始新块
                    current_chunk = {
                        "id": f"chunk_{chunk_id}",
                        "page": page_num,
                        "content": [{
                            "type": seg_type,
                            "content": seg_content
                        }],
                        "types": {seg_type},
                        "relations": []
                    }
            
            # 保存最后一个块
            if current_chunk['content']:
                semantic_chunks.append(current_chunk)
                chunk_id += 1
        
        return semantic_chunks
    
    def multimodal_vectorization_and_storage(self, semantic_chunks: List[Dict[str, Any]], embeddings: object, vs_path: str = None) -> Tuple[str, List[str]]:
        """
        多模态向量化与存储：将语义块转换为向量表示并存储到向量数据库
        
        Args:
            semantic_chunks: 语义块列表
            embeddings: 文本嵌入模型
            vs_path: 向量存储路径
            
        Returns:
            (向量存储路径, 加载的文件列表)
        """
        from datetime import datetime
        
        documents = []
        loaded_files = []
        
        for chunk in semantic_chunks:
            # 处理每个语义块
            chunk_text = ""
            chunk_images = []
            
            for item in chunk['content']:
                if item['type'] == 'text':
                    # 提取纯文本内容
                    soup = BeautifulSoup(item['content'], 'html.parser')
                    text = soup.get_text()
                    chunk_text += text + "\n"
                elif item['type'] == 'image':
                    # 提取图片路径
                    soup = BeautifulSoup(item['content'], 'html.parser')
                    img_tag = soup.find('img')
                    if img_tag:
                        img_src = img_tag.get('src')
                        # 找到完整的图片路径
                        for root, _, files in os.walk(self.temp_dir):
                            if img_src in files:
                                full_path = os.path.join(root, img_src)
                                chunk_images.append(full_path)
                                loaded_files.append(full_path)
                                break
                elif item['type'] == 'table':
                    # 提取表格文本表示
                    soup = BeautifulSoup(item['content'], 'html.parser')
                    table = soup.find('table')
                    if table:
                        # 转换为简洁的文本表示
                        table_text = "表格内容:\n"
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['th', 'td'])
                            cell_texts = [cell.get_text().strip() for cell in cells]
                            table_text += " | ".join(cell_texts) + "\n"
                        chunk_text += table_text + "\n"
                    
                    # 提取标题和说明
                    title = ""
                    caption = ""
                    title_tag = soup.find('h3', class_='segment-title')
                    if title_tag:
                        title = title_tag.get_text().strip()
                    caption_tag = soup.find('p', class_='segment-caption')
                    if caption_tag:
                        caption = caption_tag.get_text().strip()
                    
                    if title:
                        chunk_text += f"表格标题: {title}\n"
                    if caption:
                        chunk_text += f"表格说明: {caption}\n"
                        
                elif item['type'] == 'image':
                    # 提取图片相关信息
                    soup = BeautifulSoup(item['content'], 'html.parser')
                    
                    # 提取标题和说明
                    title = ""
                    caption = ""
                    title_tag = soup.find('h3', class_='segment-title')
                    if title_tag:
                        title = title_tag.get_text().strip()
                    caption_tag = soup.find('p', class_='segment-caption')
                    if caption_tag:
                        caption = caption_tag.get_text().strip()
                    
                    if title:
                        chunk_text += f"图片标题: {title}\n"
                    if caption:
                        chunk_text += f"图片说明: {caption}\n"
                
                elif item['type'] == 'formula':
                    # 提取公式文本表示
                    soup = BeautifulSoup(item['content'], 'html.parser')
                    formula_text = soup.get_text().strip()
                    
                    # 提取标题和说明
                    title = ""
                    caption = ""
                    title_tag = soup.find('h3', class_='segment-title')
                    if title_tag:
                        title = title_tag.get_text().strip()
                    caption_tag = soup.find('p', class_='segment-caption')
                    if caption_tag:
                        caption = caption_tag.get_text().strip()
                    
                    chunk_text += f"公式内容: {formula_text}\n"
                    if title:
                        chunk_text += f"公式标题: {title}\n"
                    if caption:
                        chunk_text += f"公式说明: {caption}\n"
            
            # 创建文档对象
            metadata = {
                "chunk_id": chunk['id'],
                "page": chunk['page'],
                "content_types": list(chunk['types']),
                "num_segments": len(chunk['content'])
            }
            
            # 处理图片向量
            if chunk_images:
                # 使用CLIP模型生成图片向量
                image_vectors = self._generate_image_vectors(chunk_images)
                metadata["image_vectors"] = image_vectors.tolist()
                metadata["image_paths"] = chunk_images
            
            # 文本向量将由langchain在存储时生成
            doc = Document(page_content=chunk_text.strip(), metadata=metadata)
            documents.append(doc)
        
        # 过滤复杂元数据
        documents = filter_complex_metadata(documents)
        
        # 存储到向量数据库
        if not vs_path:
            vs_path = os.path.join("vector_stores", f"MULTIMODAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vs_path
        )
        
        print(f"多模态向量生成成功，存储路径: {vs_path}")
        return vs_path, loaded_files
    
    def _generate_image_vectors(self, image_paths: List[str]) -> torch.Tensor:
        """
        使用CLIP模型生成图片向量
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            图片向量张量 (batch_size, embedding_dim)
        """
        images = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {e}")
        
        if not images:
            return torch.empty(0, 512)  # 返回空张量
        
        # 使用CLIP处理器预处理图片
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        # 生成向量
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            # 归一化向量
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        
        return outputs.cpu()
    
    def process_document(self, file_path: str, embeddings: object, vs_path: str = None) -> Tuple[str, List[str]]:
        """
        完整的多模态文档处理流程
        
        Args:
            file_path: PDF文件路径
            embeddings: 文本嵌入模型
            vs_path: 向量存储路径
            
        Returns:
            (向量存储路径, 加载的文件列表)
        """
        print(f"开始处理多模态文档: {file_path}")
        
        # 1. 结构保留的文档分割
        print("1. 执行结构保留的文档分割...")
        structured_segments = self.structure_preserving_document_split(file_path)
        print(f"   分割完成，共 {len(structured_segments)} 页")
        
        # 2. 模态特定内容提取
        print("2. 执行模态特定内容提取...")
        processed_segments = self.modality_specific_content_extraction(structured_segments)
        
        # 3. 关系保留的HTML转换
        print("3. 执行关系保留的HTML转换...")
        html_content = self.relationship_preserving_html_conversion(processed_segments)
        # 保存HTML文件用于调试
        with open(os.path.join(self.temp_dir, "processed_document.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        print("   HTML转换完成")
        
        # 4. 关系保留的语义分块
        print("4. 执行关系保留的语义分块...")
        semantic_chunks = self.relationship_preserving_semantic_chunking(html_content)
        print(f"   分块完成，共 {len(semantic_chunks)} 个语义块")
        
        # 5. 多模态向量化与存储
        print("5. 执行多模态向量化与存储...")
        vs_path, loaded_files = self.multimodal_vectorization_and_storage(semantic_chunks, embeddings, vs_path)
        print("   向量化与存储完成")
        
        print(f"多模态文档处理完成，向量存储路径: {vs_path}")
        return vs_path, loaded_files
    
    def clean_temp_files(self):
        """
        清理临时文件
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"已清理临时目录: {self.temp_dir}")


class MultimodalRAG:
    """
    多模态RAG主类，用于查询多模态向量数据库
    """
    
    def __init__(self, filepath: str, vs_path: str = None, embeddings: object = None, init: bool = True):
        self.filepath = filepath
        self.embeddings = embeddings
        self.top_k = VECTOR_SEARCH_TOP_K_PDF if hasattr(VECTOR_SEARCH_TOP_K_PDF, '__call__') else 5
        self.processor = MultimodalDocumentProcessor()
        
        if init and embeddings:
            self.vs_path, self.loaded_files = self.processor.process_document(filepath, embeddings, vs_path)
        else:
            self.vs_path = vs_path
            self.loaded_files = []
        
        # 初始化向量存储
        if self.vs_path and embeddings:
            self.vector_store = Chroma(persist_directory=self.vs_path, embedding_function=embeddings)
        else:
            self.vector_store = None
    
    def query_knowledge(self, query: str, top_k: int = None) -> List[Document]:
        """
        查询多模态知识库
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        
        actual_top_k = top_k or self.top_k
        
        # 文本查询
        docs = self.vector_store.similarity_search(query, k=actual_top_k)
        
        return docs
    
    def multimodal_query(self, query: str, query_type: str = "text", top_k: int = None) -> List[Document]:
        """
        多模态查询，支持查询PDF中已提取的图片、表格和公式
        
        Args:
            query: 查询文本
            query_type: 查询类型，可选值：text（文本）、image（图片内容查询）、table（表格内容查询）、formula（公式内容查询）
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        
        actual_top_k = top_k or self.top_k
        
        # 根据查询类型构建查询条件
        # 1. 先进行文本查询，获取相关文档
        docs = self.vector_store.similarity_search(query, k=actual_top_k * 2)  # 获取更多结果以便过滤
        
        # 2. 根据查询类型过滤结果
        if query_type in ["image", "table", "formula"]:
            filtered_docs = []
            for doc in docs:
                # 检查文档元数据中的内容类型
                content_types = doc.metadata.get("content_types", [])
                if query_type in content_types:
                    filtered_docs.append(doc)
                    if len(filtered_docs) >= actual_top_k:
                        break
            
            # 如果过滤后的结果不足，使用原始结果补充
            if len(filtered_docs) < actual_top_k:
                for doc in docs:
                    if doc not in filtered_docs:
                        filtered_docs.append(doc)
                        if len(filtered_docs) >= actual_top_k:
                            break
            
            return filtered_docs
        else:
            # 默认返回文本查询结果
            return docs[:actual_top_k]
    
    def close(self):
        """
        关闭资源，清理临时文件
        """
        self.processor.clean_temp_files()