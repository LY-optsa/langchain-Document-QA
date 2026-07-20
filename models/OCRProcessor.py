import cv2
import numpy as np
import pytesseract
import re
import string
from typing import Dict, List, Union, Optional, Tuple, Callable
from collections import Counter


class OCRProcessor:
    """
    OCR图像处理器，用于图像预处理和文字提取
    主要功能：
    1. 图像旋转校正 - 支持霍夫变换和Tesseract OSD检测
    2. 黑暗环境图像增强 - 支持多种增强算法和自动检测
    3. 文字检测和提取 - 支持文本区域检测和高精度OCR
    4. 可视化和结果导出 - 支持文本检测结果可视化
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        初始化OCR处理器
        
        Args:
            tesseract_cmd: Tesseract OCR的安装路径，如果已添加到系统PATH中则无需指定
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            加载的图像（BGR格式）
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        return image
    
    def preprocess_image(self, image: np.ndarray, 
                        grayscale: bool = True, 
                        denoise: bool = True, 
                        denoise_method: str = 'gaussian',
                        adaptive_threshold: bool = True,
                        threshold_block_size: int = 11,
                        threshold_C: int = 2,
                        morphology: bool = True,
                        morphology_kernel: Tuple[int, int] = (2, 2),
                        sharpen: bool = True,
                        resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        增强版图像预处理，优化噪声过滤和干扰抑制
        
        Args:
            image: 输入图像
            grayscale: 是否转换为灰度图
            denoise: 是否进行降噪处理
            denoise_method: 降噪方法，可选 'gaussian', 'median', 'bilateral', 'nlm'
            adaptive_threshold: 是否应用自适应阈值
            threshold_block_size: 自适应阈值的块大小（必须为奇数）
            threshold_C: 自适应阈值的常数（从平均值或高斯加权平均值中减去）
            morphology: 是否应用形态学操作
            morphology_kernel: 形态学操作的内核大小
            sharpen: 是否应用图像锐化
            resize: 调整图像大小的目标尺寸 (width, height)
            
        Returns:
            预处理后的图像
        """
        # 创建图像副本以避免修改原图
        processed_img = image.copy()
        
        # 调整大小
        if resize:
            processed_img = cv2.resize(processed_img, resize, interpolation=cv2.INTER_LANCZOS4)
        
        # 转换为灰度图
        if grayscale:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        # 多方法降噪
        if denoise and grayscale:
            if denoise_method == 'gaussian':
                # 高斯模糊 - 基础降噪
                processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
            elif denoise_method == 'median':
                # 中值滤波 - 更适合椒盐噪声
                processed_img = cv2.medianBlur(processed_img, 5)
            elif denoise_method == 'bilateral':
                # 双边滤波 - 保留边缘的降噪
                processed_img = cv2.bilateralFilter(processed_img, 9, 75, 75)
            elif denoise_method == 'nlm':
                # 非局部均值去噪 - 高级降噪方法
                processed_img = cv2.fastNlMeansDenoising(processed_img, h=10)
            else:
                # 默认使用高斯模糊
                processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
        
        # 自适应阈值 - 优化参数
        if adaptive_threshold and grayscale:
            # 确保block_size为奇数
            if threshold_block_size % 2 == 0:
                threshold_block_size += 1
            
            # 根据图像内容动态选择阈值方法
            # 先尝试Otsu阈值判断图像对比度
            _, otsu_thresh = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 计算图像的对比度
            img_mean = np.mean(processed_img)
            img_std = np.std(processed_img)
            
            # 对比度低的图像使用Gaussian_C可能效果更好，对比度高的使用Mean_C
            if img_std < 50:  # 低对比度
                processed_img = cv2.adaptiveThreshold(
                    processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, threshold_block_size, threshold_C
                )
            else:  # 高对比度
                processed_img = cv2.adaptiveThreshold(
                    processed_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, threshold_block_size, threshold_C
                )
        
        # 形态学操作 - 清理干扰
        if morphology and grayscale:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morphology_kernel)
            # 先进行闭操作填充小的空洞
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)
            # 再进行开操作去除小的噪点
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)
        
        # 图像锐化 - 增强文本边缘
        if sharpen and grayscale:
            # 使用拉普拉斯锐化
            laplacian = cv2.Laplacian(processed_img, cv2.CV_64F)
            sharpened = processed_img - 0.5 * laplacian
            processed_img = np.uint8(np.clip(sharpened, 0, 255))
        
        return processed_img
    
    def detect_rotation_angle(self, image: np.ndarray, method: str = 'hough') -> float:
        """
        检测图像的旋转角度
        
        Args:
            image: 输入图像（灰度图）
            method: 检测方法，可选 'hough' 或 'tesseract'
            
        Returns:
            检测到的旋转角度（度）
        """
        if method == 'hough':
            # 使用霍夫变换检测直线
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # 使用霍夫变换检测直线
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi/180, threshold=100,
                minLineLength=100, maxLineGap=10
            )
            
            if lines is None:
                return 0.0
            
            # 计算每条直线的角度
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # 将角度归一化到-45到45度范围
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                angles.append(angle)
            
            # 找出最常见的角度（使用直方图或Counter）
            angle_counter = Counter(angles)
            most_common_angle, _ = angle_counter.most_common(1)[0]
            
            return most_common_angle
        
        elif method == 'tesseract':
            # 使用Tesseract的OSD（方向和脚本检测）功能
            try:
                osd_data = pytesseract.image_to_osd(image)
                for line in osd_data.split('\n'):
                    if 'Rotate' in line:
                        rotation = int(line.split(':')[-1].strip())
                        return float(rotation)
            except Exception as e:
                print(f"Tesseract方向检测失败: {e}")
            return 0.0
        
        else:
            raise ValueError(f"不支持的检测方法: {method}")
    
    def correct_rotation(self, image: np.ndarray, angle: Optional[float] = None, 
                         method: str = 'hough', expand: bool = True) -> np.ndarray:
        """
        校正图像旋转
        
        Args:
            image: 输入图像
            angle: 旋转角度，如果为None则自动检测
            method: 角度检测方法，仅在angle为None时使用
            expand: 是否扩展图像以包含整个旋转后的图像
            
        Returns:
            校正后的图像
        """
        # 转换为灰度图（如果需要）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 自动检测旋转角度
        if angle is None:
            angle = self.detect_rotation_angle(gray, method)
        
        # 获取图像尺寸
        (h, w) = image.shape[:2]
        
        # 计算旋转中心点
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if expand:
            # 计算旋转后的图像尺寸
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            
            # 计算新的图像尺寸
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # 调整旋转矩阵以确保整个图像都在视野内
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # 应用仿射变换
            corrected = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        else:
            # 直接应用仿射变换
            corrected = cv2.warpAffine(image, rotation_matrix, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return corrected
    
    def enhance_dark_image(self, image: np.ndarray, method: str = 'auto', 
                          brightness: float = 1.5, contrast: float = 1.5,
                          gamma: float = 0.7, clahe_clip_limit: float = 2.0,
                          clahe_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        增强黑暗环境下的图像
        
        Args:
            image: 输入图像
            method: 增强方法，可选 'auto', 'brightness_contrast', 'histogram_equalization', 
                    'clahe', 'gamma', 'combination'
            brightness: 亮度调整系数（1.0为原始亮度）
            contrast: 对比度调整系数（1.0为原始对比度）
            gamma: 伽马校正系数（小于1.0会使图像变亮）
            clahe_clip_limit: CLAHE的裁剪限制
            clahe_grid_size: CLAHE的网格大小
            
        Returns:
            增强后的图像
        """
        # 创建图像副本
        enhanced = image.copy()
        
        # 自动检测是否需要增强（基于图像亮度）
        if method == 'auto':
            # 计算图像亮度的平均值
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            avg_brightness = np.mean(gray)
            
            # 如果平均亮度低于100，则认为是暗图像，应用组合增强
            if avg_brightness < 100:
                method = 'combination'
            else:
                return image  # 不需要增强
        
        if method == 'brightness_contrast':
            # 亮度和对比度调整
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness * 10 - 10)
        
        elif method == 'histogram_equalization':
            # 直方图均衡化
            if len(image.shape) == 3:
                # 转换到YUV色彩空间
                yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
                # 对亮度通道进行均衡化
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                # 转换回BGR
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                enhanced = cv2.equalizeHist(enhanced)
        
        elif method == 'clahe':
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
            if len(image.shape) == 3:
                # 转换到LAB色彩空间
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                # 对L通道应用CLAHE
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                # 转换回BGR
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                enhanced = clahe.apply(enhanced)
        
        elif method == 'gamma':
            # 伽马校正
            # 构建伽马查找表
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            # 应用查找表
            enhanced = cv2.LUT(enhanced, table)
        
        elif method == 'combination':
            # 组合多种增强方法
            # 1. 先进行伽马校正
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
            
            # 2. 应用CLAHE
            if len(enhanced.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
                enhanced = clahe.apply(enhanced)
            
            # 3. 调整亮度和对比度
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness * 10 - 10)
        
        else:
            raise ValueError(f"不支持的增强方法: {method}")
        
        return enhanced
    
    def dehaze(self, image: np.ndarray, omega: float = 0.95, t0: float = 0.1) -> np.ndarray:
        """
        简单的暗通道先验去雾算法，也可用于增强低光照条件下的图像
        
        Args:
            image: 输入图像
            omega: 大气光估计参数
            t0: 最小透射率
            
        Returns:
            去雾/增强后的图像
        """
        # 将图像转换到浮点类型
        img_float = image.astype(np.float64) / 255.0
        
        # 计算暗通道
        min_channel = np.min(img_float, axis=2)
        
        # 估计大气光
        # 取暗通道中最亮的0.1%的像素的平均值
        flat_min = min_channel.flatten()
        flat_min.sort()
        top_idx = int(0.001 * len(flat_min))
        A = np.mean(flat_min[-top_idx:])
        
        # 估计透射率
        transmission = 1 - omega * min_channel
        
        # 限制最小透射率
        transmission = np.maximum(transmission, t0)
        
        # 恢复图像
        dehazed = np.zeros_like(img_float)
        for i in range(3):
            dehazed[:, :, i] = (img_float[:, :, i] - A) / transmission + A
        
        # 裁剪到有效范围
        dehazed = np.clip(dehazed, 0, 1)
        
        # 转换回uint8
        dehazed = (dehazed * 255).astype(np.uint8)
        
        return dehazed
    
    def detect_text_regions(self, image: np.ndarray, min_area: int = 100, 
                          max_area_ratio: float = 0.5,
                          min_aspect_ratio: float = 0.2,
                          max_aspect_ratio: float = 10.0,
                          min_fill_ratio: float = 0.2,
                          merge_lines: bool = True,
                          line_merge_threshold: int = 10,
                          use_char_features: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        改进的文本区域检测算法，减少非文本区域的误检
        
        Args:
            image: 输入图像（最好是预处理后的二值图）
            min_area: 最小文本区域面积
            max_area_ratio: 最大文本区域面积与图像面积的比例
            min_aspect_ratio: 最小宽高比
            max_aspect_ratio: 最大宽高比
            min_fill_ratio: 最小填充率（轮廓面积与边界框面积的比值）
            merge_lines: 是否合并文本行
            line_merge_threshold: 文本行合并的垂直距离阈值
            use_char_features: 是否使用字符特征进行额外筛选
            
        Returns:
            文本区域的边界框列表 [(x, y, width, height), ...]
        """
        # 如果是彩色图像，转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 确保图像是二值图
        if len(np.unique(gray)) > 2:
            # 使用Otsu阈值，但添加预处理步骤
            # 先进行高斯模糊以减少噪声影响
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary = cv2.bitwise_not(gray)  # 确保文本为白色，背景为黑色
        
        # 计算图像面积
        img_area = gray.shape[0] * gray.shape[1]
        max_area = int(img_area * max_area_ratio)
        
        # 形态学操作 - 增强文本连通性
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选文本区域
        text_regions = []
        for i, contour in enumerate(contours):
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 基本面积筛选
            if area < min_area or area > max_area:
                continue
            
            # 宽高比筛选
            aspect_ratio = w / h
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            # 填充率筛选（轮廓面积与边界框面积的比值）
            contour_area = cv2.contourArea(contour)
            fill_ratio = contour_area / area
            if fill_ratio < min_fill_ratio:
                continue
            
            # 字符特征筛选
            if use_char_features:
                # 计算轮廓的周长
                perimeter = cv2.arcLength(contour, True)
                
                # 计算形状复杂度（圆度）
                if perimeter > 0:
                    circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                    # 文本通常不是完美的圆形，圆度较低
                    if circularity > 0.8:  # 太圆的可能是噪声或非文本元素
                        continue
                
                # 检查是否是独立区域（不是其他区域的子区域）
                if hierarchy is not None and hierarchy[0][i][3] != -1:
                    # 这是一个子区域，可能是字符内部的空洞
                    # 可以根据情况决定是否保留
                    parent_idx = hierarchy[0][i][3]
                    parent_contour = contours[parent_idx]
                    _, _, parent_w, parent_h = cv2.boundingRect(parent_contour)
                    
                    # 如果子区域太大，可能不是字符内部的空洞
                    if w > parent_w * 0.5 or h > parent_h * 0.5:
                        continue
            
            # 检查文本密度（二值图中白色像素的比例）
            roi = binary[y:y+h, x:x+w]
            if roi.size > 0:
                text_density = np.sum(roi == 255) / roi.size
                # 文本区域通常有一定的密度，太低可能是噪声
                if text_density < 0.1 or text_density > 0.9:
                    continue
            
            text_regions.append((x, y, w, h))
        
        # 合并文本行
        if merge_lines and text_regions:
            text_regions = self._merge_text_lines(text_regions, line_merge_threshold)
        
        # 按x坐标排序（从左到右），然后按y坐标排序（从上到下）
        text_regions.sort(key=lambda rect: (rect[1], rect[0]))
        
        # 进一步验证文本区域
        validated_regions = []
        for region in text_regions:
            if self._validate_text_region(binary, region):
                validated_regions.append(region)
        
        return validated_regions
    
    def _merge_text_lines(self, regions: List[Tuple[int, int, int, int]], 
                         threshold: int = 10) -> List[Tuple[int, int, int, int]]:
        """
        合并属于同一文本行的区域
        
        Args:
            regions: 检测到的文本区域列表
            threshold: 垂直距离阈值
            
        Returns:
            合并后的文本区域列表
        """
        if not regions:
            return []
        
        # 按y坐标排序
        sorted_regions = sorted(regions, key=lambda r: r[1])
        merged = [sorted_regions[0]]
        
        for current in sorted_regions[1:]:
            last = merged[-1]
            
            # 计算两个区域的垂直中心距离
            last_center_y = last[1] + last[3] // 2
            current_center_y = current[1] + current[3] // 2
            vertical_distance = abs(last_center_y - current_center_y)
            
            # 如果垂直距离小于阈值，认为它们在同一行
            if vertical_distance < threshold:
                # 合并两个区域
                new_x = min(last[0], current[0])
                new_y = min(last[1], current[1])
                new_width = max(last[0] + last[2], current[0] + current[2]) - new_x
                new_height = max(last[1] + last[3], current[1] + current[3]) - new_y
                merged[-1] = (new_x, new_y, new_width, new_height)
            else:
                merged.append(current)
        
        return merged
    
    def _validate_text_region(self, binary: np.ndarray, region: Tuple[int, int, int, int]) -> bool:
        """
        验证区域是否为有效文本区域
        
        Args:
            binary: 二值化图像
            region: 要验证的区域 (x, y, w, h)
            
        Returns:
            是否为有效文本区域
        """
        x, y, w, h = region
        
        # 确保区域在图像范围内
        h_img, w_img = binary.shape
        if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
            return False
        
        # 获取区域ROI
        roi = binary[y:y+h, x:x+w]
        
        # 计算水平投影
        horizontal_projection = np.sum(roi, axis=1) / 255
        # 计算垂直投影
        vertical_projection = np.sum(roi, axis=0) / 255
        
        # 文本区域通常有较多的水平变化
        h_non_zero = np.count_nonzero(horizontal_projection)
        v_non_zero = np.count_nonzero(vertical_projection)
        
        # 文本区域的水平投影非零比例不应太低
        if h_non_zero / h < 0.3:
            return False
        
        # 文本区域的垂直投影非零比例也不应太低
        if v_non_zero / w < 0.1:
            return False
        
        # 检查是否有连续的空白区域（文本通常不会有太大的空白区域）
        max_blank_h = 0
        current_blank = 0
        for val in horizontal_projection:
            if val == 0:
                current_blank += 1
                max_blank_h = max(max_blank_h, current_blank)
            else:
                current_blank = 0
        
        # 如果有过大的水平空白区域，可能不是文本
        if max_blank_h > h * 0.5:
            return False
        
        return True
    
    def extract_text(self, image: np.ndarray, lang: str = 'chi_sim+eng', 
                    config: str = '', 
                    config_template: str = 'default',
                    psm: int = 6,  # 页面分割模式
                    oem: int = 3,  # OCR引擎模式
                    whitelist: str = '',  # 白名单字符
                    blacklist: str = '',  # 黑名单字符
                    preprocess: bool = True, 
                    preprocess_params: Optional[Dict] = None,
                    auto_rotate: bool = True, 
                    rotate_method: str = 'tesseract',
                    enhance_dark: bool = True,
                    enhance_method: str = 'auto',
                    min_confidence: int = 0) -> Dict[str, Union[str, List[Dict]]]:
        """
        优化的文本提取函数，支持多种Tesseract配置参数
        
        Args:
            image: 输入图像
            lang: OCR语言，默认为中文简体+英文
            config: 自定义Tesseract配置参数
            config_template: 配置模板，可选 'default', 'text', 'table', 'document', 'digits_only', 'quiet'
            psm: 页面分割模式 (0-13)
                0: 仅方向和脚本检测
                6: 假设为单个统一块文本（默认）
                7: 假设为单行文本
                8: 假设为单个词
                11: 稀疏文本，查找尽可能多的文本
                12: 稀疏文本，带OSD
                13: 原始行，旁路Tesseract页面分割
            oem: OCR引擎模式 (0-3)
                0: 仅传统引擎
                1: 仅LSTM引擎
                2: 传统引擎和LSTM引擎
                3: 默认，根据可用性选择（默认）
            whitelist: 白名单字符，只识别这些字符
            blacklist: 黑名单字符，不识别这些字符
            preprocess: 是否进行预处理
            preprocess_params: 预处理参数，会传递给preprocess_image方法
            auto_rotate: 是否自动校正旋转
            rotate_method: 旋转检测方法，'hough'或'tesseract'
            enhance_dark: 是否自动增强暗图像
            enhance_method: 暗图像增强方法
            min_confidence: 最小置信度阈值，低于此值的文本块将被过滤
            
        Returns:
            包含提取文字和详细信息的字典
        """
        # 创建图像副本
        processed_img = image.copy()
        
        # 自动增强暗图像
        if enhance_dark:
            processed_img = self.enhance_dark_image(processed_img, method=enhance_method)
        
        # 自动校正旋转
        if auto_rotate:
            processed_img = self.correct_rotation(processed_img, method=rotate_method)
        
        # 预处理
        if preprocess:
            if preprocess_params is None:
                # 使用优化的默认参数
                preprocess_params = {
                    'denoise_method': 'bilateral',  # 使用双边滤波保留边缘
                    'threshold_block_size': 15,     # 稍大的块大小以适应不同文本大小
                    'threshold_C': 3,               # 调整阈值常数
                    'morphology': True,             # 启用形态学操作
                    'sharpen': True                 # 启用锐化
                }
            processed_img = self.preprocess_image(processed_img, **preprocess_params)
        
        # 构建Tesseract配置参数
        tess_config = self._build_tesseract_config(config_template, psm, oem, whitelist, blacklist, config)
        
        # 提取全文本
        full_text = pytesseract.image_to_string(processed_img, lang=lang, config=tess_config)
        
        # 提取详细的文字数据（包括位置信息）
        data = pytesseract.image_to_data(processed_img, lang=lang, config=tess_config, output_type=pytesseract.Output.DICT)
        
        # 整理文字块信息
        text_blocks = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            confidence = int(data['conf'][i])
            # 过滤低置信度的文本块
            if confidence > min_confidence:
                text = data['text'][i].strip()
                if text:  # 只保留非空文本
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text_blocks.append({
                        'text': text,
                        'position': {'x': x, 'y': y, 'width': w, 'height': h},
                        'confidence': confidence,
                        'page_num': data['page_num'][i],
                        'block_num': data['block_num'][i],
                        'par_num': data['par_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i]
                    })
        
        return {
            'full_text': full_text,
            'text_blocks': text_blocks,
            'num_blocks': len(text_blocks),
            'config_used': tess_config
        }
    
    def _build_tesseract_config(self, template: str = 'default', psm: int = 6, 
                              oem: int = 3, whitelist: str = '', 
                              blacklist: str = '', custom_config: str = '') -> str:
        """
        构建Tesseract配置参数
        
        Args:
            template: 配置模板
            psm: 页面分割模式
            oem: OCR引擎模式
            whitelist: 白名单字符
            blacklist: 黑名单字符
            custom_config: 自定义配置
            
        Returns:
            完整的Tesseract配置字符串
        """
        # 基础配置
        config_parts = [f'--oem {oem}', f'--psm {psm}']
        
        # 模板特定配置
        if template == 'text':
            # 纯文本优化配置
            config_parts.extend([
                '-c preserve_interword_spaces=1',        # 保留单词间空格
                '-c textord_space_size_is_variable=1',   # 考虑可变空格大小
                '-c textord_tabfind_find_tables=0'       # 禁用表格检测
            ])
        elif template == 'table':
            # 表格优化配置
            config_parts.extend([
                '-c textord_tabfind_find_tables=1',      # 启用表格检测
                '-c tessedit_create_hocr=1',             # 创建HOCR输出以保留表格结构
                '-c preserve_interword_spaces=1'         # 保留单词间空格
            ])
        elif template == 'document':
            # 文档优化配置
            config_parts.extend([
                '-c preserve_interword_spaces=1',        # 保留单词间空格
                '-c textord_space_size_is_variable=1',   # 考虑可变空格大小
                '-c language_model_penalty_non_freq_dict_word=0.5',  # 降低非词典词惩罚
                '-c language_model_penalty_garbage=1'    # 增加垃圾字符惩罚
            ])
        elif template == 'digits_only':
            # 仅数字优化配置
            config_parts.extend([
                '-c tessedit_char_whitelist=0123456789.'  # 仅识别数字和小数点
            ])
        elif template == 'quiet':
            # 安静模式，减少输出
            config_parts.extend(['-c enable_new_segsearch=0'])
        
        # 自定义白名单和黑名单
        if whitelist:
            config_parts.append(f'-c tessedit_char_whitelist={whitelist}')
        if blacklist:
            config_parts.append(f'-c tessedit_char_blacklist={blacklist}')
        
        # 添加自定义配置
        if custom_config:
            config_parts.append(custom_config)
        
        # 合并所有配置
        return ' '.join(config_parts)
    
    def postprocess_text(self, ocr_result: Dict[str, Union[str, List[Dict]]],
                        confidence_threshold: int = 60,
                        min_text_length: int = 1,
                        max_text_length: int = 1000,
                        remove_whitespace: bool = False,
                        normalize_text: bool = True,
                        keywords_include: Optional[List[str]] = None,
                        keywords_exclude: Optional[List[str]] = None,
                        merge_adjacent_blocks: bool = True,
                        custom_filter: Optional[Callable[[Dict], bool]] = None) -> Dict[str, Union[str, List[Dict]]]:
        """
        对OCR识别结果进行后处理，过滤不合理的识别结果
        
        Args:
            ocr_result: OCR识别结果字典
            confidence_threshold: 置信度阈值，低于此值的文本块将被过滤
            min_text_length: 最小文本长度，短于此长度的文本块将被过滤
            max_text_length: 最大文本长度，长于此长度的文本块将被过滤
            remove_whitespace: 是否移除所有空白字符
            normalize_text: 是否规范化文本
            keywords_include: 关键词白名单，只有包含这些关键词的文本块才会保留
            keywords_exclude: 关键词黑名单，包含这些关键词的文本块将被过滤
            merge_adjacent_blocks: 是否合并相邻的文本块
            custom_filter: 自定义过滤函数，接收文本块字典，返回True保留，False过滤
            
        Returns:
            后处理后的OCR结果
        """
        text_blocks = ocr_result.get('text_blocks', [])
        
        # 1. 按置信度过滤
        filtered_blocks = [block for block in text_blocks if block['confidence'] >= confidence_threshold]
        
        # 2. 按文本长度过滤
        filtered_blocks = [block for block in filtered_blocks 
                          if min_text_length <= len(block['text']) <= max_text_length]
        
        # 3. 文本特征过滤（过滤乱码和不合理字符）
        filtered_blocks = [block for block in filtered_blocks if self._is_valid_text(block['text'])]
        
        # 4. 关键词过滤
        if keywords_include:
            filtered_blocks = [block for block in filtered_blocks 
                              if any(keyword.lower() in block['text'].lower() for keyword in keywords_include)]
        
        if keywords_exclude:
            filtered_blocks = [block for block in filtered_blocks 
                              if not any(keyword.lower() in block['text'].lower() for keyword in keywords_exclude)]
        
        # 5. 自定义过滤
        if custom_filter:
            filtered_blocks = [block for block in filtered_blocks if custom_filter(block)]
        
        # 6. 合并相邻文本块
        if merge_adjacent_blocks and len(filtered_blocks) > 1:
            filtered_blocks = self._merge_adjacent_text_blocks(filtered_blocks)
        
        # 7. 文本规范化
        if normalize_text:
            for block in filtered_blocks:
                block['text'] = self._normalize_text(block['text'], remove_whitespace)
        
        # 8. 重新生成全文本
        full_text = ' '.join([block['text'] for block in filtered_blocks])
        
        return {
            'full_text': full_text,
            'text_blocks': filtered_blocks,
            'num_blocks': len(filtered_blocks),
            'filtered_count': len(text_blocks) - len(filtered_blocks)
        }
    
    def _is_valid_text(self, text: str) -> bool:
        """
        检查文本是否合理，过滤乱码和异常字符
        
        Args:
            text: 要检查的文本
            
        Returns:
            文本是否有效的布尔值
        """
        # 空字符串检查
        if not text or not text.strip():
            return False
        
        # 检查特殊字符比例
        special_chars = sum(1 for c in text if c in string.punctuation) / len(text)
        if special_chars > 0.8:  # 如果80%以上都是特殊字符，可能是乱码
            return False
        
        # 检查纯数字（如果需要过滤纯数字的话）
        # if text.isdigit() and len(text) > 10:  # 过长的纯数字可能是噪点
        #     return False
        
        # 检查控制字符
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
        if control_chars > 0:  # 包含控制字符
            return False
        
        # 检查重复字符
        if len(set(text)) / len(text) < 0.2:  # 字符多样性低于20%
            return False
        
        # 检查中文字符比例（对于中文文档）
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > 0:  # 如果有中文字符
            # 确保至少有一定比例的中文字符
            if chinese_chars / len(text) < 0.1 and len(text) > 3:  # 如果文本长度大于3且中文字符少于10%
                return False
        
        return True
    
    def _normalize_text(self, text: str, remove_whitespace: bool = False) -> str:
        """
        规范化文本，去除多余空格和特殊字符
        
        Args:
            text: 要规范化的文本
            remove_whitespace: 是否移除所有空白字符
            
        Returns:
            规范化后的文本
        """
        if remove_whitespace:
            # 移除所有空白字符
            text = re.sub(r'\s+', '', text)
        else:
            # 标准化空白字符（多个空格替换为单个空格）
            text = re.sub(r'\s+', ' ', text)
        
        # 移除控制字符（除了换行符和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def _merge_adjacent_text_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        合并相邻的文本块，处理被错误分割的文本
        
        Args:
            text_blocks: 文本块列表
            
        Returns:
            合并后的文本块列表
        """
        if not text_blocks:
            return []
        
        # 按照位置排序
        sorted_blocks = sorted(text_blocks, key=lambda x: (x['position']['y'], x['position']['x']))
        merged_blocks = [sorted_blocks[0]]
        
        for current in sorted_blocks[1:]:
            previous = merged_blocks[-1]
            
            # 检查是否应该合并（位置接近且在同一行）
            prev_y = previous['position']['y']
            prev_h = previous['position']['height']
            prev_x = previous['position']['x'] + previous['position']['width']
            curr_y = current['position']['y']
            curr_x = current['position']['x']
            
            # 垂直方向重叠或接近，且水平方向相邻
            y_overlap = abs(curr_y - prev_y) < prev_h * 0.3
            x_distance = curr_x - prev_x
            
            if y_overlap and 0 <= x_distance < prev_h * 0.5:
                # 合并文本块
                merged_text = previous['text'] + (' ' if x_distance > 0 else '') + current['text']
                merged_confidence = min(previous['confidence'], current['confidence'])  # 取较低置信度
                
                merged_position = {
                    'x': previous['position']['x'],
                    'y': min(previous['position']['y'], current['position']['y']),
                    'width': curr_x + current['position']['width'] - previous['position']['x'],
                    'height': max(previous['position']['y'] + previous['position']['height'], 
                                 current['position']['y'] + current['position']['height']) - \
                             min(previous['position']['y'], current['position']['y'])
                }
                
                merged_blocks[-1] = {
                    'text': merged_text,
                    'position': merged_position,
                    'confidence': merged_confidence,
                    'page_num': previous['page_num'],
                    'block_num': previous['block_num'],
                    'is_merged': True  # 标记为合并块
                }
            else:
                merged_blocks.append(current)
        
        return merged_blocks
    
    def visualize_text_detection(self, image: np.ndarray, text_blocks: List[Dict], 
                               show_confidence: bool = True, 
                               color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        可视化文本检测结果
        
        Args:
            image: 输入图像
            text_blocks: 文本块信息列表
            show_confidence: 是否显示置信度
            color: 边界框颜色 (B, G, R)
            
        Returns:
            可视化后的图像
        """
        # 创建图像副本
        vis_img = image.copy()
        
        # 确保图像是彩色的
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # 绘制边界框和文本
        for block in text_blocks:
            pos = block['position']
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            
            # 绘制边界框
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # 绘制文本和置信度
            text_to_show = block['text']
            if show_confidence and 'confidence' in block:
                text_to_show += f" (C:{block['confidence']})"
            
            # 确保文本在图像范围内
            text_y = max(y - 10, 20)
            cv2.putText(vis_img, text_to_show, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_img
    
    def save_result_image(self, image: np.ndarray, output_path: str) -> bool:
        """
        保存处理后的图像
        
        Args:
            image: 要保存的图像
            output_path: 输出文件路径
            
        Returns:
            是否保存成功
        """
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            print(f"保存图像失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    """
    OCRProcessor 使用示例
    
    注意：使用前请确保已安装以下依赖：
    1. opencv-python
    2. numpy
    3. pytesseract
    4. Tesseract OCR引擎（需要单独安装）
    
    在Windows上，可能需要设置Tesseract的安装路径
    """
    
    # 创建OCR处理器实例
    # 如果Tesseract未添加到系统PATH，需要指定路径
    # processor = OCRProcessor(tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe')
    processor = OCRProcessor()
    
    # 示例1: 基本OCR处理流程
    def example_basic_ocr(image_path, output_image_path=None):
        """基本OCR处理流程示例"""
        print("\n=== 基本OCR处理示例 ===")
        
        # 加载图像
        image = processor.load_image(image_path)
        print(f"加载图像成功，尺寸: {image.shape}")
        
        # 执行OCR提取文字（自动处理旋转和暗图像增强）
        result = processor.extract_text(
            image,
            lang='chi_sim+eng',  # 中英文识别
            preprocess=True,
            auto_rotate=True,
            enhance_dark=True
        )
        
        # 输出结果
        print(f"检测到 {result['num_blocks']} 个文本块")
        print("\n提取的文本:")
        print(result['full_text'])
        
        # 可视化检测结果
        if result['text_blocks'] and output_image_path:
            vis_image = processor.visualize_text_detection(image, result['text_blocks'])
            processor.save_result_image(vis_image, output_image_path)
            print(f"\n可视化结果已保存到: {output_image_path}")
        
        return result
    
    # 示例2: 旋转图像校正
    def example_rotation_correction(image_path):
        """旋转图像校正示例"""
        print("\n=== 旋转图像校正示例 ===")
        
        # 加载图像
        image = processor.load_image(image_path)
        
        # 方法1: 使用霍夫变换检测并校正旋转
        corrected_hough = processor.correct_rotation(image, method='hough')
        print("使用霍夫变换完成旋转校正")
        
        # 方法2: 使用Tesseract OSD检测并校正旋转
        corrected_tesseract = processor.correct_rotation(image, method='tesseract')
        print("使用Tesseract OSD完成旋转校正")
        
        return corrected_hough, corrected_tesseract
    
    # 示例3: 暗图像增强
    def example_dark_image_enhancement(image_path):
        """暗图像增强示例"""
        print("\n=== 暗图像增强示例 ===")
        
        # 加载图像
        image = processor.load_image(image_path)
        
        # 方法1: 自动检测和增强
        auto_enhanced = processor.enhance_dark_image(image, method='auto')
        print("完成自动图像增强")
        
        # 方法2: 亮度对比度调整
        bc_enhanced = processor.enhance_dark_image(
            image, 
            method='brightness_contrast',
            brightness=2.0,  # 增加亮度
            contrast=1.5     # 增加对比度
        )
        print("完成亮度对比度调整")
        
        # 方法3: 组合增强（伽马校正 + CLAHE + 亮度对比度）
        combined_enhanced = processor.enhance_dark_image(image, method='combination')
        print("完成组合增强")
        
        # 方法4: 去雾增强（也适用于低光照条件）
        dehazed = processor.dehaze(image)
        print("完成去雾/增强")
        
        return auto_enhanced, bc_enhanced, combined_enhanced, dehazed
    
    # 示例4: 文本区域检测
    def example_text_region_detection(image_path):
        """文本区域检测示例"""
        print("\n=== 文本区域检测示例 ===")
        
        # 加载图像
        image = processor.load_image(image_path)
        
        # 预处理图像（二值化有助于文本检测）
        processed = processor.preprocess_image(image, adaptive_threshold=True)
        
        # 检测文本区域
        regions = processor.detect_text_regions(processed, min_area=50, max_area_ratio=0.6)
        print(f"检测到 {len(regions)} 个文本区域")
        
        # 绘制检测到的区域
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        for i, (x, y, w, h) in enumerate(regions):
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis_image, f"Region {i+1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return vis_image, regions


    def test_ocr_optimization(image_path: str):
        """
        测试OCR优化效果，比较优化前后的结果
        
        Args:
            image_path: 测试图像路径
        """
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        
        # 创建OCR处理器实例
        ocr_processor = OCRProcessor()
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            return
        
        print("="*80)
        print(f"开始测试OCR优化效果: {image_path}")
        print("="*80)
        
        # 测试1: 基本OCR（无优化）
        print("\n1. 测试基本OCR（无优化）:")
        basic_result = ocr_processor.extract_text(
            image,
            preprocess=False,
            auto_rotate=False,
            enhance_dark=False,
            config='--oem 3 --psm 6'
        )
        
        # 测试2: 仅优化预处理
        print("\n2. 测试优化预处理:")
        preprocess_result = ocr_processor.extract_text(
            image,
            preprocess=True,
            preprocess_params={
                'denoise_method': 'bilateral',
                'threshold_block_size': 15,
                'threshold_C': 3,
                'morphology': True,
                'sharpen': True
            },
            auto_rotate=False,
            enhance_dark=False,
            config='--oem 3 --psm 6'
        )
        
        # 测试3: 仅优化Tesseract配置
        print("\n3. 测试优化Tesseract配置:")
        config_result = ocr_processor.extract_text(
            image,
            preprocess=False,
            auto_rotate=False,
            enhance_dark=False,
            config_template='document',
            psm=6
        )
        
        # 测试4: 完整优化（预处理+旋转校正+暗环境增强+Tesseract优化）
        print("\n4. 测试完整优化:")
        full_result = ocr_processor.extract_text(
            image,
            preprocess=True,
            preprocess_params={
                'denoise_method': 'bilateral',
                'threshold_block_size': 15,
                'threshold_C': 3,
                'morphology': True,
                'sharpen': True
            },
            auto_rotate=True,
            rotate_method='tesseract',
            enhance_dark=True,
            enhance_method='auto',
            config_template='document',
            psm=6
        )
        
        # 测试5: 完整优化 + 后处理
        print("\n5. 测试完整优化 + 后处理:")
        postprocessed_result = ocr_processor.postprocess_text(
            full_result,
            confidence_threshold=60,
            min_text_length=2,
            merge_adjacent_blocks=True,
            normalize_text=True
        )
        
        # 生成测试报告
        print("\n" + "="*80)
        print("OCR优化效果对比报告")
        print("="*80)
        print(f"基本OCR: {len(basic_result['text_blocks'])}个文本块")
        print(f"优化预处理: {len(preprocess_result['text_blocks'])}个文本块")
        print(f"优化Tesseract配置: {len(config_result['text_blocks'])}个文本块")
        print(f"完整优化: {len(full_result['text_blocks'])}个文本块")
        print(f"完整优化+后处理: {len(postprocessed_result['text_blocks'])}个文本块")
        print(f"后处理过滤掉: {postprocessed_result['filtered_count']}个不合理文本块")
        
        # 计算平均置信度
        def avg_confidence(text_blocks):
            if not text_blocks:
                return 0
            return sum(block['confidence'] for block in text_blocks) / len(text_blocks)
        
        print("\n平均置信度对比:")
        print(f"基本OCR: {avg_confidence(basic_result['text_blocks']):.1f}")
        print(f"优化预处理: {avg_confidence(preprocess_result['text_blocks']):.1f}")
        print(f"优化Tesseract配置: {avg_confidence(config_result['text_blocks']):.1f}")
        print(f"完整优化: {avg_confidence(full_result['text_blocks']):.1f}")
        print(f"完整优化+后处理: {avg_confidence(postprocessed_result['text_blocks']):.1f}")
        
        # 保存各阶段处理后的图像用于视觉比较
        # 预处理图像
        preprocessed_img = ocr_processor.preprocess_image(
            image.copy(),
            denoise_method='bilateral',
            threshold_block_size=15,
            threshold_C=3,
            morphology=True,
            sharpen=True
        )
        cv2.imwrite('test_preprocessed.jpg', preprocessed_img)
        
        # 校正后的图像
        corrected_img = ocr_processor.correct_rotation(preprocessed_img, method='tesseract')
        cv2.imwrite('test_corrected.jpg', corrected_img)
        
        # 可视化最终识别结果
        vis_image = ocr_processor.visualize_text_detection(image.copy(), postprocessed_result['text_blocks'])
        cv2.imwrite('test_final_result.jpg', vis_image)
        
        print("\n测试图像已保存:")
        print("- test_preprocessed.jpg: 预处理后的图像")
        print("- test_corrected.jpg: 校正后的图像")
        print("- test_final_result.jpg: 最终识别结果可视化")
        print("\n测试完成！")


    def batch_test_optimization(image_paths: List[str]):
        """
        批量测试OCR优化效果
        
        Args:
            image_paths: 测试图像路径列表
        """
        print("开始批量测试OCR优化效果")
        print("="*80)
        
        overall_stats = {
            'basic': [],
            'preprocess': [],
            'config': [],
            'full': [],
            'postprocessed': []
        }
        
        for image_path in image_paths:
            # 运行单次测试
            test_ocr_optimization(image_path)
            
            # 为了演示，这里只是模拟收集统计数据
            # 在实际应用中，你可以修改test_ocr_optimization函数返回结果
            # 然后在这里汇总统计
        
        print("\n" + "="*80)
        print("批量测试完成")
        print("="*80)


    def custom_filter_example(text_block: Dict) -> bool:
        """
        自定义过滤函数示例
        
        Args:
            text_block: 文本块字典
            
        Returns:
            是否保留该文本块
        """
        # 示例1: 过滤掉包含特定模式的文本（如日期格式）
        # date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        # if re.search(date_pattern, text_block['text']):
        #     return False
        
        # 示例2: 过滤掉纯符号或数字
        if text_block['text'].isdigit() and len(text_block['text']) > 8:
            return False
        
        # 示例3: 保留特定区域的文本
        position = text_block['position']
        # 假设我们只关心图像中间区域的文本
        # if position['x'] < 100 or position['x'] > 500:  # 根据实际图像尺寸调整
        #     return False
        
        return True


    def advanced_ocr_pipeline_example(image_path: str):
        """
        高级OCR处理流程示例，集成所有优化技术
        
        Args:
            image_path: 图像路径
        """
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
        
        # 创建OCR处理器实例
        ocr_processor = OCRProcessor()
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            return
        
        # 1. 智能预处理 - 根据图像类型自动选择参数
        # 计算图像亮度以判断是否为暗图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        avg_brightness = np.mean(gray)
        is_dark = avg_brightness < 100
        
        # 根据图像亮度动态调整预处理参数
        preprocess_params = {
            'denoise_method': 'bilateral',  # 双边滤波保留边缘
            'morphology': True,             # 启用形态学操作
            'sharpen': True                 # 启用锐化
        }
        
        # 暗图像使用不同的阈值参数
        if is_dark:
            preprocess_params['threshold_block_size'] = 21  # 更大的块大小
            preprocess_params['threshold_C'] = 5            # 更大的常数
        else:
            preprocess_params['threshold_block_size'] = 15
            preprocess_params['threshold_C'] = 3
        
        # 2. 执行OCR识别
        ocr_result = ocr_processor.extract_text(
            image,
            lang='chi_sim+eng',
            preprocess=True,
            preprocess_params=preprocess_params,
            auto_rotate=True,
            rotate_method='tesseract',
            enhance_dark=is_dark,  # 只在需要时进行暗图像增强
            enhance_method='histogram_equalization' if is_dark else 'auto',
            config_template='document',
            psm=6  # 假设为单个统一块文本
        )
        
        # 3. 智能后处理 - 多层次过滤
        # 定义业务相关关键词（根据实际应用场景调整）
        business_keywords = ['合同', '协议', '条款', '甲方', '乙方', '签名', '日期']
        
        # 垃圾文本关键词
        garbage_keywords = ['页眉', '页脚', '页码', '水印']
        
        postprocessed_result = ocr_processor.postprocess_text(
            ocr_result,
            confidence_threshold=70,  # 较高的置信度阈值
            min_text_length=2,       # 过滤单字符
            keywords_include=business_keywords if business_keywords else None,
            keywords_exclude=garbage_keywords,
            merge_adjacent_blocks=True,
            custom_filter=custom_filter_example  # 应用自定义过滤函数
        )
        
        # 4. 输出优化结果
        print(f"原始OCR结果: {len(ocr_result['text_blocks'])}个文本块")
        print(f"优化后结果: {len(postprocessed_result['text_blocks'])}个文本块")
        print(f"过滤掉的无效块: {postprocessed_result['filtered_count']}")
        
        # 5. 可视化结果
        vis_image = ocr_processor.visualize_text_detection(image.copy(), postprocessed_result['text_blocks'])
        
        # 显示结果
        cv2.imshow('Advanced OCR Results', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return postprocessed_result