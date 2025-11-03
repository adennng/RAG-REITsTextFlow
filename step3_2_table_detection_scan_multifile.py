#扫描页跨页表格检测
import os
import cv2
import torch
import json
import base64
import shutil
import logging
import numpy as np
from PIL import Image
from openai import OpenAI
from transformers import TableTransformerForObjectDetection
import torchvision.transforms as transforms
import re
from builtins import open
from step3_cross_page_table_detector import CrossPageTableDetector
from model_config import MODEL_CONFIG
from file_paths_config import OUTPUT_DIR, table_transformer_path  # 从配置文件中导入文件路径配置
import pytesseract  # [新增] 用于检测文字方向
from pytesseract import TesseractError
import pymysql
from db_config import get_db_announcement_config  # 新增数据库配置导入

# 启用离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 设置Tesseract数据路径
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract/tessdata"

# ====================
# 配置区
# ====================
class Config:
    # 模型参数
    table_transformer_path = table_transformer_path  # 从配置文件导入
    table_transformer_confidence = 0.4
    min_table_area = 10000
    
    # 跨页判断阈值
    cross_page_top_threshold = 200
    cross_page_bottom_threshold = 100
    # [新增] 针对旋转页面的跨页判断阈值
    rotated_cross_page_left_threshold = 100
    rotated_cross_page_right_threshold = 100
   
    # 连续页最小数量
    min_continuous_pages = 2
    
    # 大模型配置
    model_provider = "zhipu"  # 模型厂商名称
    model_name = "GLM-4V-Flash"     # 模型名称
    glm_timeout = 60
    glm_max_retry = 5
    log_level = logging.DEBUG

# ====================
# 初始化日志（修改后）
# ====================
# 创建日志目录，使用相对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "table_processing_scan_multifile.log")

class HttpLogFilter(logging.Filter):
    def filter(self, record):
        suppress_messages = [
            'Request options:',
            'Sending HTTP Request:',
            'send_request_headers',
            'send_request_body',
            'receive_response_headers',
            'receive_response_body',
            'response_closed',
            'HTTP Response:',
            'request_id:'
        ]
        if record.levelno == logging.DEBUG:
            return not any(msg in record.getMessage() for msg in suppress_messages)
        return True

class SpecialLogFilter(logging.Filter):
    def filter(self, record):
        # 允许所有WARNING及以上级别的日志（除了"空图片目录"）
        if record.levelno >= logging.WARNING:
            if "空图片目录" in record.getMessage():
                return False
            return True
        
        # 允许特定的INFO消息
        message = record.getMessage()
        allowed_info_messages = [
            "没有找到需要处理的文件。",
            "扫描表格检测全部完成！状态已更新至数据库。",
            "处理结果统计：",
            "总待处理文件:",
            "成功处理文件:",
            "处理失败文件:",
            "剩余未处理文件:",
            "失败文件详情："
        ]
        return any(msg in message for msg in allowed_info_messages)

file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.addFilter(SpecialLogFilter())
file_handler.addFilter(HttpLogFilter())
stream_handler = logging.StreamHandler()
stream_handler.setLevel(Config.log_level)
stream_handler.addFilter(HttpLogFilter())

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def get_pending_files_from_db():
    """
    从数据库获取需要进行扫描表格检测的文件
    条件: text_extracted='true' AND table_detection_scan_done='false' AND doc_type_1 != '无关'
    返回: 按fund_code分组的文件列表字典
    """
    db_conf = get_db_announcement_config()
    conn = None
    try:
        conn = pymysql.connect(**db_conf)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        sql = """
        SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
               doc_type_1, doc_type_2, announcement_link
        FROM processed_files 
        WHERE text_extracted='true' 
          AND table_detection_scan_done='false' 
          AND doc_type_1 != '无关'
        ORDER BY fund_code, file_name
        """
        
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # 按fund_code分组
        grouped_files = {}
        for row in results:
            fund_code = row['fund_code']
            if fund_code not in grouped_files:
                grouped_files[fund_code] = []
            grouped_files[fund_code].append(row)
        
        return grouped_files
        
    except Exception as e:
        if conn:
            conn.close()
        logging.error(f"数据库查询失败: {e}")
        raise e

def update_database_status(file_name):
    """更新数据库中的table_detection_scan_done状态"""
    try:
        db_config = get_db_announcement_config()
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            sql = """UPDATE processed_files 
                    SET table_detection_scan_done = %s 
                    WHERE file_name = %s"""
            cursor.execute(sql, ("true", file_name))  # 使用字符串 "true"
        connection.commit()
        logging.info(f"数据库状态更新成功: {file_name}")
        return True
    except Exception as e:
        logging.error(f"数据库更新失败: {file_name} - {str(e)}")
        return False
    finally:
        if connection:
            connection.close()

# ====================
# 辅助函数：解决中文路径读取问题
# ====================
def cv2_imread_unicode(img_path):
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        image = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"cv2_imread_unicode 读取失败: {img_path}, 错误: {str(e)}")
        return None

# ====================
# 加载TableTransformer模型
# ====================
def load_models():
    try:
        logging.info("Loading TableTransformer model...")
        model = TableTransformerForObjectDetection.from_pretrained(
            Config.table_transformer_path,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
        return model
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        raise

# [新增] 文字方向检测函数
def detect_text_orientation(img_path):
    """
    如果检测到 0° 或 180° → 'vertical'
    如果检测到 90° 或 270° → 'rotated'
    """
    try:
        img = Image.open(img_path)
    except Exception as e:
        logging.error(f"打开图片失败: {img_path}, 错误: {e}")
        return 'vertical'  # 默认竖直
    try:
        osd_data = pytesseract.image_to_osd(img, config='--psm 0')
    except pytesseract.TesseractError as te:
        logging.warning(f"Tesseract OSD 出错，返回默认vertical: {te}")
        return 'vertical'

    match = re.search(r'Orientation in degrees: (\d+)', osd_data)
    orientation_degrees = 0
    if match:
        orientation_degrees = int(match.group(1))
        logging.debug(f"Tesseract 检测到 {img_path} 文字方向: {orientation_degrees}°")
    else:
        logging.warning(f"无法解析 Tesseract OSD 输出，默认 0°")

    # 统一：只要是 90° 或 270° 都视为 rotated（逆时针 90）
    if orientation_degrees in [90, 270]:
        return 'rotated'
    else:
        return 'vertical'

# ===== 两个函数：分别计算竖直页面和旋转页面时的文字边距 =====
def extract_text_distances_vertical(image_path):
    """
    对于竖直页面：计算文字到页面顶部/底部的距离
    返回 (text_top_dist, text_bottom_dist)
    """
    img = cv2_imread_unicode(image_path)
    if img is None:
        return None, None

    height, width = img.shape[:2]
    # 裁剪左侧区域
    left_img = img[0:height, 0:int(width * 0.6)]
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(threshold_img, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_distance = height
    bottom_distance = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:
            if y < top_distance:
                top_distance = y
            if (y + h) > bottom_distance:
                bottom_distance = y + h

    if top_distance == height and bottom_distance == 0:
        return None, None

    text_top_dist = top_distance
    text_bottom_dist = height - bottom_distance
    return text_top_dist, text_bottom_dist


def extract_text_distances_rotated(image_path):
    """
    对于旋转页面：我们把“页面的左边”视为“表格的上边”，
    所以这里计算文字到页面左/右边的距离。
    返回 (text_left_dist, text_right_dist)
    """
    img = cv2_imread_unicode(image_path)
    if img is None:
        return None, None

    height, width = img.shape[:2]
    # 同样可以裁剪上侧 60% 或别的区域，这里为了演示就裁剪上半部分
    # 也可以改成裁剪“页面上方 60%”
    # 不过一般旋转后文字方向是竖排，所以要根据实际情况找更合理的裁剪方向

    # 这里演示：裁剪“页面上方 100%高、左侧 60%宽” → 仍可行，但要注意文字可能竖着排
    left_img = img[0:height, 0:int(width * 0.6)]
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(threshold_img, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_distance = width
    right_distance = 0

    # 注意这里 x, y, w, h 的含义
    # x是距离页面左边的距离；如果要“把页面的左边看成上边”，实际上得做更多旋转思考
    # 这里仅作为演示
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:
            # 最小的x就是“最左”
            if x < left_distance:
                left_distance = x
            if (x + w) > right_distance:
                right_distance = (x + w)

    if left_distance == width and right_distance == 0:
        return None, None

    text_left_dist = left_distance
    text_right_dist = width - right_distance
    return text_left_dist, text_right_dist
    
# ====================
# 表格检测模块
# ====================
class TableDetector:
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect_tables(self, img_path):
        try:
            image = Image.open(img_path)
        except Exception as e:
            logging.error(f"图片打开失败: {str(e)}")
            return []
        pixel_values = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(pixel_values)
            # 转换边界框
        id2label = self.model.config.id2label
        id2label[len(id2label)] = "no object"
        detected = self._parse_outputs(outputs, image.size, id2label)
        # 过滤结果
        valid_boxes = []
        for obj in detected:
            if obj['score'] > Config.table_transformer_confidence:
                x1, y1, x2, y2 = obj['bbox']
                if (x2 - x1) * (y2 - y1) > Config.min_table_area:
                    valid_boxes.append((x1, y1, x2, y2))
        logging.info(f"检测到表格数量: {len(valid_boxes)}，坐标: {valid_boxes}")
        return valid_boxes

    def _parse_outputs(self, outputs, img_size, id2label):
        """解析模型输出"""
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self._rescale_bboxes(pred_bboxes, img_size)]
        return [
            {"label": id2label[int(label)], "score": score, "bbox": bbox}
            for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes)
            if id2label[int(label)] != "no object"
        ]

    def _rescale_bboxes(self, boxes, size):
        """重新缩放边界框"""
        width, height = size
        boxes = self._box_cxcywh_to_xyxy(boxes)
        boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        return boxes

    def _box_cxcywh_to_xyxy(self, x):
        """坐标格式转换"""
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack([
            (x_c - 0.5 * w),
            (y_c - 0.5 * h),
            (x_c + 0.5 * w),
            (y_c + 0.5 * h)
        ], dim=1)

# ====================
# 大模型处理模块（修改后）
# ====================
class ModelProcessor:
    def __init__(self):
        # 查找匹配的模型配置
        self.model_config = MODEL_CONFIG.get(Config.model_provider, {}).get(Config.model_name, None)
        if not self.model_config:
            raise ValueError(f"未找到模型配置: 厂商={Config.model_provider}, 模型={Config.model_name}")
        self.client = OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["base_url"]
        )
        self.prompt = ("请分析图片并返回以下JSON信息：\n"
                       "as_table: 是否存在表格（true/false）\n"
                       "请注意，只返回has_table，其余内容都不要提供。")

    def process_image(self, img_path):
        base64_image = self._encode_image(img_path)
        for _ in range(Config.glm_max_retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    timeout=Config.glm_timeout
                )
                return self._parse_response(response.choices[0].message.content)
            except Exception as e:
                logging.warning(f"API调用失败: {str(e)}")
        return None

    def _encode_image(self, img_path):
        """Base64编码图片"""
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _parse_response(self, response):
        """
        如果返回纯 true/false，就当做 {"has_table": true/false}。
        如果返回 JSON 字符串，就照常解析。
        """
        try:
            cleaned = re.sub(r'```json|```|\n|\\', '', response).strip()
            logging.debug(f"大模型返回的原始内容: {cleaned}")

            # 如果只是一串 "true" 或 "false"
            if cleaned.lower() in ("true", "false"):
                bool_val = (cleaned.lower() == "true")
                return {"has_table": bool_val, "page_info": {"top_clear": False, "bottom_clear": False}}

            # 否则就当做 JSON
            data = json.loads(cleaned)
            return {
                "has_table": data.get("has_table", False),
                "page_info": data.get("page_info", {"top_clear": False, "bottom_clear": False})
            }
        except Exception as e:
            logging.error(f"解析失败: {str(e)}")
            return None


# ====================
def extract_text_distances(image_path, orientation='vertical'):
    """
    如果 orientation='vertical'，提取文字上/下距离。
    如果 orientation='rotated'，同样先旋转图像 90 度再测量“上/下”（对原图其实是左/右）。
    """
    img = cv2_imread_unicode(image_path)
    if img is None:
        logging.warning(f"无法读取图片: {image_path}")
        return None, None

    if orientation == 'rotated':
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    height, width = img.shape[:2]
    left_img = img[0:height, 0:int(width * 0.6)]
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(threshold_img, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    top_distance = height
    bottom_distance = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:
            if y < top_distance:
                top_distance = y
            if (y + h) > bottom_distance:
                bottom_distance = y + h

    if top_distance == height and bottom_distance == 0:
        logging.debug(f"未检测到文字: {image_path}")
        return None, None

    text_top_dist = top_distance
    text_bottom_dist = height - bottom_distance
    return text_top_dist, text_bottom_dist


# ====================
# 跨页处理模块（主要逻辑不变，但将输出目录参数化）
# ====================
class PageMerger:
    @staticmethod
    def find_continuous_pages(processed_pages):
        """识别连续跨页表格"""
        sequences = []
        current_seq = []
        for i, page in enumerate(processed_pages):
            img_name = page['file']
            if page['has_table']:
                if not current_seq:
                    current_seq.append(i)
                    logging.debug(f"开始新序列 @ {img_name}")
                else:
                    prev_page = processed_pages[current_seq[-1]]
                    if PageMerger._is_continuous(prev_page, page):
                        current_seq.append(i)
                        logging.debug(f"追加连续页 @ {img_name}")
                    else:
                        logging.debug(f"连续性中断 @ {img_name}")
                        PageMerger._finalize_sequence(current_seq, sequences, processed_pages)
                        current_seq = [i]
            else:
                PageMerger._finalize_sequence(current_seq, sequences, processed_pages)
                current_seq = []
        PageMerger._finalize_sequence(current_seq, sequences, processed_pages)
        return [seq for seq in sequences if len(seq) >= Config.min_continuous_pages]

    @staticmethod
    def _finalize_sequence(current_seq, sequences, processed_pages):
        if len(current_seq) >= Config.min_continuous_pages:
            sequences.append(current_seq.copy())
            img_names = [processed_pages[i]['file'] for i in current_seq]
            logging.info(f"发现有效跨页序列: {img_names}")

    @staticmethod
    def _is_continuous(prev_page, curr_page):
        """
        区分 'vertical' / 'rotated'，并用不同的差值和阈值做判断。
        """
        if not prev_page['has_table'] or not curr_page['has_table']:
            return False

        # 如果 orientation 不同，则不拼接
        if prev_page['orientation'] != curr_page['orientation']:
            logging.debug(f"方向不同，无法跨页拼接: {prev_page['file']} -> {curr_page['file']}")
            return False

        orient = prev_page['orientation']  # 或 curr_page['orientation']，相同就行

        if orient == 'vertical':
            # 竖直页面
            p_text_bottom = prev_page.get('text_bottom_dist')
            p_table_bottom = prev_page.get('table_bottom_dist')
            c_text_top = curr_page.get('text_top_dist')
            c_table_top = curr_page.get('table_top_dist')
            if None in [p_text_bottom, p_table_bottom, c_text_top, c_table_top]:
                return False

            diff_prev_bottom = abs(p_table_bottom - p_text_bottom)
            diff_curr_top = abs(c_table_top - c_text_top)
            logging.debug(f"上一页表格-文字底部差: {diff_prev_bottom}px")
            logging.debug(f"当前页表格-文字顶部差: {diff_curr_top}px")

            condition_prev = diff_prev_bottom < Config.cross_page_bottom_threshold
            condition_curr = diff_curr_top < Config.cross_page_top_threshold
            is_cross = condition_prev and condition_curr
            logging.debug(
                f"竖直拼接判断: {prev_page['file']} -> {curr_page['file']} => {is_cross}"
            )
            return is_cross

        else:
            # orient == 'rotated'
            # 旋转页面
            p_text_right = prev_page.get('text_right_dist')
            p_table_right = prev_page.get('table_right_dist')
            c_text_left = curr_page.get('text_left_dist')
            c_table_left = curr_page.get('table_left_dist')
            if None in [p_text_right, p_table_right, c_text_left, c_table_left]:
                return False

            diff_prev_right = abs(p_table_right - p_text_right)
            diff_curr_left = abs(c_table_left - c_text_left)
            logging.debug(f"上一页表格-文字右侧差: {diff_prev_right}px")
            logging.debug(f"当前页表格-文字左侧差: {diff_curr_left}px")

            condition_prev = diff_prev_right < Config.rotated_cross_page_right_threshold
            condition_curr = diff_curr_left < Config.rotated_cross_page_left_threshold
            is_cross = condition_prev and condition_curr
            logging.debug(
                f"旋转拼接判断: {prev_page['file']} -> {curr_page['file']} => {is_cross}"
            )
            return is_cross


    @staticmethod
    def merge_pages(processed_pages, sequence, output_dir):
        """
        根据 orientation 决定是上下拼接还是左右拼接。
        假设整条 sequence 中 orientation 都相同，因此可以取第一个页面的 orientation 即可。
        """
        try:
            if not sequence:
                return False

            orientation = processed_pages[sequence[0]]['orientation']
            images = []
            merged_name_parts = []

            for idx_i, idx in enumerate(sequence):
                page = processed_pages[idx]
                img_path = os.path.join(output_dir, page['file'])
                file_stem, _ = os.path.splitext(page['file'])
                if file_stem.startswith("page_"):
                    number_part = file_stem.replace("page_", "")
                    merged_name_parts.append(number_part)
                else:
                    merged_name_parts.append(file_stem)

                with Image.open(img_path) as img:
                    if orientation == 'vertical':
                        # 获取表格上下边界
                        table_top = min(t['top'] for t in page['tables'])
                        table_bottom = max(t['bottom'] for t in page['tables'])

                        if idx_i == 0:
                            crop_top = 0
                            crop_bottom = table_bottom
                        elif idx_i == len(sequence) - 1:
                            crop_top = table_top
                            crop_bottom = img.height
                        else:
                            crop_top = table_top
                            crop_bottom = min(table_bottom, img.height)

                        cropped = img.crop((0, crop_top, img.width, crop_bottom))
                        images.append(cropped)

                    else:
                        # orientation == 'rotated'
                        # 获取表格左右边界
                        table_left = min(t['left'] for t in page['tables'])
                        table_right = max(t['right'] for t in page['tables'])

                        if idx_i == 0:
                            crop_left = 0
                            crop_right = table_right
                        elif idx_i == len(sequence) - 1:
                            crop_left = table_left
                            crop_right = img.width
                        else:
                            crop_left = table_left
                            crop_right = min(table_right, img.width)

                        cropped = img.crop((crop_left, 0, crop_right, img.height))
                        images.append(cropped)

            # 拼接
            if orientation == 'vertical':
                # 垂直拼接
                total_height = sum(img.height for img in images)
                merged_img = Image.new('RGB', (images[0].width, total_height))
                y_offset = 0
                for m_img in images:
                    merged_img.paste(m_img, (0, y_offset))
                    y_offset += m_img.height
            else:
                # 水平拼接
                total_width = sum(img.width for img in images)
                merged_img = Image.new('RGB', (total_width, images[0].height))
                x_offset = 0
                for m_img in images:
                    merged_img.paste(m_img, (x_offset, 0))
                    x_offset += m_img.width

            final_merged_name = "page_" + "-".join(merged_name_parts) + ".png"
            output_path = os.path.join(output_dir, final_merged_name)
            merged_img.save(output_path)
            logging.info(f"已保存拼接结果到: {output_path}")

            # 删除原文件
            for idx in sequence:
                output_file = os.path.join(output_dir, processed_pages[idx]['file'])
                if os.path.exists(output_file):
                    os.remove(output_file)
                    logging.debug(f"已删除原文件: {output_file}")
            return True
        except Exception as e:
            logging.error(f"拼接失败: {str(e)}", exc_info=True)
            return False


# ====================
# 多文件处理核心：处理单个PDF（新增临时文件删除逻辑）
# ====================

def process_single_pdf(pdf_info, model, detector, model_processor):
    """
    处理单个PDF的扫描页表格检测及跨页拼接
    从 pdf_info 中获取 PDF 对应输出文件夹，
    然后在该文件夹下的 temp_pdf_images 目录中收集扫描页图片，
    将检测到表格的图片复制到 table_image 文件夹，
    对同一文件内连续的扫描页执行跨页拼接，
    在数据库和 processed_files.json 中标记该文件已完成扫描页表格检测。
    最后在temp_pdf_images删除对应的图片。
    """
    try:  # 新增异常捕获
        fund_code = pdf_info.get("fund_code", "")
        pdf_folder_name = os.path.splitext(os.path.basename(pdf_info["file_path"]))[0]

        # 定位基金文件夹
        fund_folder_name = ""
        for folder in os.listdir(OUTPUT_DIR):
            if folder.startswith(fund_code):
                fund_folder_name = folder
                break
        if not fund_folder_name:
            error_msg = f"未找到对应基金文件夹，基金代码: {fund_code}"
            logging.error(error_msg)
            return False, error_msg  # 修改返回值

        pdf_folder_dir = os.path.join(OUTPUT_DIR, fund_folder_name, pdf_folder_name)
        if not os.path.exists(pdf_folder_dir):
            error_msg = f"未找到对应PDF文件夹: {pdf_folder_dir}"
            logging.error(error_msg)
            return False, error_msg  # 修改返回值

        input_img_dir = os.path.join(pdf_folder_dir, "temp_pdf_images")
        table_img_dir = os.path.join(pdf_folder_dir, "table_image")
        os.makedirs(table_img_dir, exist_ok=True)

        processed_pages = []
        img_files = sorted(
            [f for f in os.listdir(input_img_dir) if f.startswith('page_') and f.endswith('.png')],
            key=lambda x: int(re.search(r'page_(\d+)', x).group(1))
        )

        # 处理空文件情况（只更新数据库状态）
        if not img_files:
            logging.debug(f"空图片目录: {input_img_dir}")  # 修改为debug级别
            # 只更新数据库
            update_database_status(pdf_info["file_name"])
            return True, None  # 返回成功状态

        logging.info(f"开始处理文件: {pdf_info['file_path']}，共 {len(img_files)} 页")

        for img_file in img_files:
            img_path = os.path.join(input_img_dir, img_file)
            if not os.path.exists(img_path):
                logging.error(f"图片文件不存在: {img_path}")
                continue
            logging.info(f"处理 {img_file}...")

            # 1) 检测文本方向
            orientation = detect_text_orientation(img_path)

            # 2) 根据 orientation 提取文字距离
            if orientation == 'vertical':
                text_top_dist, text_bottom_dist = extract_text_distances_vertical(img_path)
                text_left_dist = None
                text_right_dist = None
            else:
                # orientation == 'rotated'
                text_left_dist, text_right_dist = extract_text_distances_rotated(img_path)
                text_top_dist = None
                text_bottom_dist = None

            # 3) 检测表格
            boxes = detector.detect_tables(img_path)
            has_table = len(boxes) > 0

            model_result = None
            if has_table:
                model_result = model_processor.process_image(img_path)
                has_table = model_result['has_table'] if model_result else False

            with Image.open(img_path) as tmp_img:
                page_width, page_height = tmp_img.size

            # 根据 orientation 来存表格“上下”或“左右”距离
            if orientation == 'vertical':
                table_top_dist = None
                table_bottom_dist = None
                table_left_dist = None
                table_right_dist = None
                if boxes:
                    # 竖直：表格顶部/底部
                    table_top_dist = min(b[1] for b in boxes)  # box: (x1,y1,x2,y2)
                    table_bottom_dist = page_height - max(b[3] for b in boxes)
            else:
                # orientation == 'rotated'
                table_top_dist = None
                table_bottom_dist = None
                table_left_dist = None
                table_right_dist = None
                if boxes:
                    # 旋转：表格左侧/右侧
                    table_left_dist = min(b[0] for b in boxes)
                    table_right_dist = page_width - max(b[2] for b in boxes)

            page_data = {
                "file": img_file,
                "has_table": has_table,
                "orientation": orientation,
                "tables": [{"left": x1, "top": y1, "right": x2, "bottom": y2} for (x1, y1, x2, y2) in boxes],
                "height": page_height,
                "width": page_width,

                # 竖直文本距离
                "text_top_dist": text_top_dist,
                "text_bottom_dist": text_bottom_dist,
                # 旋转文本距离
                "text_left_dist": text_left_dist,
                "text_right_dist": text_right_dist,

                # 竖直表格距离
                "table_top_dist": table_top_dist,
                "table_bottom_dist": table_bottom_dist,
                # 旋转表格距离
                "table_left_dist": table_left_dist,
                "table_right_dist": table_right_dist,
            }
            processed_pages.append(page_data)

            if has_table:
                dst_path = os.path.join(table_img_dir, img_file)
                shutil.copy(img_path, dst_path)
                logging.info(f"已保存表格图片: {dst_path}")

        sequences = PageMerger.find_continuous_pages(processed_pages)
        logging.info(f"发现跨页序列: {sequences}")

        for seq in sequences:
            PageMerger.merge_pages(processed_pages, seq, table_img_dir)

        checker = SupplementaryChecker(table_img_dir, input_img_dir)
        for page_data in processed_pages:
            if checker.process_page(page_data, processed_pages):
                logging.info(f"成功处理补充拼接：{page_data['file']}")

        # ========== 只更新数据库状态 ==========
        db_success = update_database_status(pdf_info["file_name"])
        if not db_success:
            error_msg = f"数据库状态更新失败: {pdf_info['file_name']}"
            logging.error(error_msg)
            return False, error_msg

        # 删除临时图片文件（原有逻辑保持不变）
        processed_images = [page['file'] for page in processed_pages if page['has_table']]
        for img_file in processed_images:
            temp_img_path = os.path.join(input_img_dir, img_file)
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                    logging.info(f"已删除临时图片文件: {temp_img_path}")
                except Exception as e:
                    logging.error(f"删除临时图片失败: {temp_img_path}, 错误: {e}")

        logging.info(f"完成处理: {pdf_info['file_path']}")
        return True, None  # 返回成功状态

    except Exception as e:
        error_msg = f"处理文件异常: {pdf_info['file_path']} - {str(e)}"
        logging.error(error_msg, exc_info=True)
        return False, error_msg  # 返回失败状态和错误信息

# ====================
# 新增补充检测模块
# ====================
class SupplementaryChecker:
    def __init__(self, output_dir, input_img_dir):
        self.output_dir = output_dir
        self.input_img_dir = input_img_dir
        self.detector = CrossPageTableDetector()

    def _parse_page_numbers(self, filename):
        numbers = []
        parts = re.findall(r'\d+', filename)
        for p in parts:
            if p.isdigit():
                numbers.append(int(p))
        return sorted(numbers)

    def _is_already_merged(self, current_page_num):
        target_pages = {current_page_num, current_page_num + 1}
        for f in os.listdir(self.output_dir):
            if not f.startswith("page_"):
                continue
            pages = set(self._parse_page_numbers(f))
            if target_pages.issubset(pages):
                logging.debug(f"发现已合并文件: {f} 包含页码 {current_page_num} 和 {current_page_num+1}")
                return True
        return False

    def _find_prev_image(self, current_file):
        standard_path = os.path.join(self.output_dir, current_file)
        if os.path.exists(standard_path):
            return standard_path
        current_page_num = re.search(r'page_(\d+)', current_file)
        if current_page_num:
            current_page_num = int(current_page_num.group(1))
        else:
            return None

        for f in os.listdir(self.output_dir):
            if f == current_file:
                continue
            pages = self._parse_page_numbers(f)
            if current_page_num in pages:
                return os.path.join(self.output_dir, f)
        return None

    def _find_next_page_image(self, current_page_num):
        next_page_num = current_page_num + 1
        candidates = []
        for f in os.listdir(self.output_dir):
            pages = self._parse_page_numbers(f)
            if pages and pages[0] == next_page_num:
                candidates.append((f, pages))
        candidates.sort(key=lambda x: (len(x[1]), x[1][0]))
        exact_match = [c for c in candidates if c[1] == [next_page_num]]
        if exact_match:
            return os.path.join(self.output_dir, exact_match[0][0])
        if candidates:
            return os.path.join(self.output_dir, candidates[0][0])
        temp_img = os.path.join(self.input_img_dir, f"page_{next_page_num}.png")
        return temp_img if os.path.exists(temp_img) else None

    def _generate_merged_name(self, prev_name, next_name):
        def extract_pages(name):
            return list(map(int, re.findall(r'\d+', os.path.basename(name))))
        prev_pages = extract_pages(prev_name)
        next_pages = extract_pages(next_name)
        merged_pages = sorted(list(set(prev_pages + next_pages)))
        return f"page_{'-'.join(map(str, merged_pages))}.png"

        # 解析页码
        prev_pages = extract_pages(prev_name)
        next_pages = extract_pages(next_name)
        all_pages = sorted(list(set(prev_pages + next_pages)))

        # 生成连续页码表达式
        parts = []
        i = 0
        while i < len(all_pages):
            start = all_pages[i]
            j = i
            while j + 1 < len(all_pages) and all_pages[j+1] == all_pages[j] + 1:
                j += 1
            if j > i:
                parts.append(f"{start}-{all_pages[j]}")
            else:
                parts.append(str(start))
            i = j + 1
        # 修改连续页码合并逻辑为不合并
        parts = list(map(str, all_pages))
        return f"page_{'-'.join(parts)}.png"
    
    def process_page(self, page_data, processed_pages):
        if not page_data['has_table']:
            return False
        current_file = page_data['file']
        match = re.search(r'page_(\d+)', current_file)
        if not match:
            return False
        current_page_num = int(match.group(1))

        if self._is_already_merged(current_page_num):
            logging.info(f"跳过 {current_file}：已存在合并文件")
            return False
            

        # 只做简单判断，不区分旋转
        table_bottom_dist = page_data.get('table_bottom_dist')
        text_bottom_dist = page_data.get('text_bottom_dist')
        # 如果你也想支持旋转这里的补充拼接，可以自己加逻辑
        if table_bottom_dist is None or text_bottom_dist is None:
            return False
        if abs(table_bottom_dist - text_bottom_dist) > Config.cross_page_bottom_threshold:
            return False

        prev_img_path = self._find_prev_image(current_file)
        if not prev_img_path:
            return False
        next_img_path = self._find_next_page_image(current_page_num)
        if not next_img_path:
            return False

        result = self.detector.check_cross_page(prev_img_path, next_img_path)
        if result['is_cross_page']:
            success = self._merge_pages(prev_img_path, next_img_path, page_data, result)
            if success:
                logging.info(f"拼接文件已保存到: {self.output_dir}")
            return success
        return False

    def _merge_pages(self, prev_path, next_path, page_data, cross_result):
        try:
            prev_img = Image.open(prev_path)
            next_img = Image.open(next_path)
            prev_crop_bottom = page_data['height'] - page_data['tables'][0]['bottom']
            next_crop_top = cross_result['horizontal_distance'] if cross_result['horizontal_distance'] is not None else 300
            prev_cropped = prev_img.crop((0, 0, prev_img.width, prev_img.height - prev_crop_bottom))
            next_cropped = next_img.crop((0, next_crop_top, next_img.width, next_img.height))

            total_height = prev_cropped.height + next_cropped.height
            merged_img = Image.new('RGB', (prev_cropped.width, total_height))
            y_offset = 0
            merged_img.paste(prev_cropped, (0, y_offset))
            y_offset += prev_cropped.height
            merged_img.paste(next_cropped, (0, y_offset))

            new_name = self._generate_merged_name(os.path.basename(prev_path), os.path.basename(next_path))
            output_path = os.path.join(self.output_dir, new_name)
            merged_img.save(output_path)
            logging.info(f"已保存到table_image: {output_path}")

            to_delete = []
            if prev_path.startswith(self.output_dir):
                to_delete.append(prev_path)
            if next_path.startswith(self.output_dir):
                to_delete.append(next_path)

            for path in to_delete:
                try:
                    os.remove(path)
                    logging.info(f"删除原文件: {os.path.relpath(path, self.output_dir)}")
                except Exception as e:
                    logging.error(f"删除失败: {str(e)}")

            return True
        except Exception as e:
            logging.error(f"合并失败: {str(e)}", exc_info=True)
            return False

# ====================
# 主流程（新增完成提示）
# ====================
def main():
    model = load_models()
    detector = TableDetector(model)
    model_processor = ModelProcessor()
    
    try:
        # 从数据库获取待处理文件
        processed_files = get_pending_files_from_db()
    except Exception as e:
        logging.error(f"从数据库获取待处理文件失败: {e}")
        print(f"从数据库获取待处理文件失败: {e}")
        return

    # 统计相关变量
    total_files = 0
    processed_count = 0
    failed_files = []

    # 计算总数
    for fund_code, pdf_infos in processed_files.items():
        total_files += len(pdf_infos)

    logging.info(f"共有 {total_files} 个文件待处理")

    for fund_code, pdf_infos in processed_files.items():
        for pdf_info in pdf_infos:
            try:
                success, error = process_single_pdf(pdf_info, model, detector, model_processor)
                if success:
                    processed_count += 1
                else:
                    failed_files.append({
                        'file_name': pdf_info['file_name'],
                        'error': error
                    })
            except Exception as e:
                failed_files.append({
                    'file_name': pdf_info['file_name'],
                    'error': str(e)
                })
                logging.error(f"处理文件异常: {pdf_info['file_name']}，错误: {str(e)}", exc_info=True)

    # 输出统计信息（新增）
    remaining = total_files - processed_count - len(failed_files)
    logging.info("="*50)
    logging.info("处理结果统计：")
    logging.info(f"- 总待处理文件: {total_files}")
    logging.info(f"- 成功处理文件: {processed_count}")
    logging.info(f"- 处理失败文件: {len(failed_files)}")
    logging.info(f"- 剩余未处理文件: {remaining}")
    logging.info("="*50)
    
    # 同时在终端也打印（保持原有功能）
    print("\n处理结果统计：")
    print(f"总待处理文件: {total_files}")
    print(f"成功处理文件: {processed_count}")
    print(f"处理失败文件: {len(failed_files)}")
    print(f"剩余未处理文件: {remaining}")

    if len(failed_files) > 0:
        logging.info("\n失败文件详情：")
        print("\n失败文件详情：")  # 终端也显示
        for f in failed_files:
            logging.info(f"文件: {f['file_name']} - 错误: {f['error']}")
            print(f"文件: {f['file_name']} - 错误: {f['error']}")  # 终端也显示

    if processed_count > 0 or len(failed_files) > 0:
        completion_msg = "扫描表格检测全部完成！状态已更新至数据库。"
        logging.info(completion_msg)
        print(completion_msg)  # 终端也显示
    else:
        no_files_msg = "没有找到需要处理的文件。"
        logging.info(no_files_msg)
        print(no_files_msg)  # 终端也显示


if __name__ == "__main__":
    main()


# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()