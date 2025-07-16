#!/usr/bin/env python3
"""
SoM处理器 - 基于OmniParser-v2.0
简洁版本，专门用于生成Set-of-Mark标注
"""

import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

# OmniParser imports
from som import MarkHelper, plot_boxes_with_marks
from utils import get_yolo_model, get_som_labeled_img, check_ocr_box

class SoMProcessor:
    def __init__(self):
        self.setup_models()
        self.mark_helper = MarkHelper()
    
    def setup_models(self):
        """下载并加载模型"""
        # 下载权重
        if not os.path.exists('./weights/icon_detect/model.pt'):
            print("📦 Downloading OmniParser weights...")
            snapshot_download(
                repo_id="microsoft/OmniParser-v2.0", 
                local_dir='./weights'
            )
        
        # 加载YOLO模型
        self.yolo_model = get_yolo_model(model_path='./weights/icon_detect/model.pt')
        print("✅ Models loaded")
    
    def process_image(self, image, box_threshold=0.01, ocr_threshold=0.9):
        """处理单张图像，返回SoM标注"""
        
        # 自适应参数
        box_overlay_ratio = image.size[0] / 3200
        draw_config = {
            # 'text_scale': 0.8 * box_overlay_ratio,
            'text_scale': 0.9,
            'text_thickness': max(int(2 * box_overlay_ratio), 2),
            'text_padding': max(int(3 * box_overlay_ratio), 2),
            'thickness': max(int(3 * box_overlay_ratio), 2),   
        }
        
        
        # OCR检测
        ocr_result, _ = check_ocr_box(
            image, 
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'paragraph': False, 'text_threshold': ocr_threshold},
            use_paddleocr=False
        )
        text, ocr_bbox = ocr_result
        
        # 生成SoM
        labeled_img_b64, coordinates, _ = get_som_labeled_img(
            image,
            self.yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_config,
            caption_model_processor=None,  # 不使用caption
            ocr_text=text,
            use_local_semantics=False,     # 跳过语义理解
            iou_threshold=0.9
        )
        
        # 解码图像
        som_image = Image.open(io.BytesIO(base64.b64decode(labeled_img_b64)))
        
        return som_image, coordinates