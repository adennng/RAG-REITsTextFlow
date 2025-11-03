# 非表格图片描述的主脚本（PaddleOCR版本） - 可复用&可独立执行

import os
import json
import time
import logging
import re
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import pymysql
import db_config
from file_paths_config import OUTPUT_DIR
from common_utils import safe_json_dump
import threading

# 在代码开头添加配置参数
TOP_CROP_PIXELS = 300    # 上边距裁剪像素（可调整）
BOTTOM_CROP_PIXELS = 300 # 下边距裁剪像素（可调整）

# 初始化PaddleOCR（最简配置）并配备线程锁，避免多线程并发导致段错误
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
# Predictor 非线程安全；串行化访问
ocr_lock = threading.Lock()

# ============ 日志只输出WARNING及以上 + 特殊两句话 =============
class SpecificLogFilter(logging.Filter):
    """
    与 LLM 脚本相同的过滤器:
    - 级别>=WARNING
    - 或者消息是 "没有找到需要处理的文件。" / "非表格图片描述全部完成！"
    """
    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if record.getMessage() in ("没有找到需要处理的文件。", "非表格图片描述全部完成！"):
            return True
        return False

# 获取脚本所在目录，确保日志文件生成在log目录下
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "not_table_detection.log")

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.addFilter(SpecificLogFilter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(file_handler)
else:
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    logger.addHandler(file_handler)
# ================================================================

def get_announcement_connection():
    """
    获取数据库 announcement 的连接
    """
    config = db_config.get_db_announcement_config()
    return pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        charset=config["charset"],
        cursorclass=pymysql.cursors.DictCursor
    )

def get_pending_files_from_db():
    """
    从数据库获取需要进行非表格图片描述的文件（PaddleOCR版本）
    条件: text_extracted='true' AND table_detection_scan_done='true' 
          AND not_table_describe_done='false' AND doc_type_1 != '无关'
    返回: 按fund_code分组的文件列表字典
    """
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = """
            SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
                   doc_type_1, doc_type_2, announcement_link, text_extracted,
                   table_detection_scan_done, not_table_describe_done
            FROM processed_files 
            WHERE text_extracted='true' 
              AND table_detection_scan_done='true' 
              AND not_table_describe_done='false' 
              AND doc_type_1 != '无关'
            ORDER BY fund_code, file_name
            """
            cursor.execute(sql)
            results = cursor.fetchall()
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
        logging.error(f"数据库查询失败: {e}")
        raise e

def crop_image(image_path, top_pixels, bottom_pixels):
    """
    裁剪图片去除页眉页脚区域
    :param image_path: 原始图片路径
    :param top_pixels: 上边距裁剪像素
    :param bottom_pixels: 下边距裁剪像素
    :return: 裁剪后的numpy数组图像
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        crop_top = max(0, top_pixels)
        crop_bottom = max(0, height - bottom_pixels)

        if crop_bottom <= crop_top:
            raise ValueError(
                f"无效裁剪参数: 顶部({crop_top}) ≥ 底部({crop_bottom}) | 原始尺寸: {width}x{height}"
            )

        cropped_img = img.crop((0, crop_top, width, crop_bottom))
        return np.array(cropped_img.convert('RGB'))

    except Exception as e:
        raise RuntimeError(f"图片裁剪失败: {str(e)}") from e

def clean_and_reorganize_text(text: str, title_max_length: int = 30) -> str:
    """
    清洗和重组OCR识别结果：
    1. 去除多余的空格和换行
    2. 合并不合理的换行符
    3. 保留疑似标题行的换行符
    """
    lines = text.split('\n')
    cleaned_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 判断是否为标题行（短文本且不以标点结尾）
        if len(line) < title_max_length and not re.search(r'[。！？…,.!?]$', line):
            cleaned_lines.append(line)
            i += 1  # 保留标题行的换行符
        else:
            # 合并连续的非标题行
            current_line = line
            while i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not next_line:
                    i += 1
                    continue
                if not re.search(r'[。！？…,.!?]$', current_line):
                    current_line += next_line
                    i += 1
                else:
                    break
            cleaned_lines.append(current_line)
            i += 1

    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = cleaned_text.replace(" ", "")
    return cleaned_text

def run_paddle_ocr_on_image(image_path: str) -> str:
    """
    对单张图片执行裁剪和 OCR。若未识别到文本则抛出 ValueError("PaddleOCR未识别到任何文本")
    """
    cropped_array = crop_image(image_path, TOP_CROP_PIXELS, BOTTOM_CROP_PIXELS)

    # 串行化调用 PaddleOCR，避免多线程竞争导致 SIGSEGV
    with ocr_lock:
        result = ocr.ocr(cropped_array)

    if not result or not result[0]:
        # 没识别到任何文本
        raise ValueError("PaddleOCR未识别到任何文本")

    lines = []
    for page in result:
        for line in page:
            if len(line) >= 2 and line[1]:
                lines.append(line[1][0])
    ocr_text = "\n".join(lines)

    cleaned_text = clean_and_reorganize_text(ocr_text)
    return cleaned_text

def update_pdf_status_in_db(file_name, status_field, status_value):
    """
    更新数据库中单个PDF文件的状态
    """
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = f"UPDATE processed_files SET {status_field}=%s WHERE file_name=%s"
            cursor.execute(sql, (status_value, file_name))
            conn.commit()
        conn.close()
        logging.info(f"已更新数据库 {file_name} 的 {status_field}={status_value}")
    except Exception as e:
        logging.error(f"更新数据库状态失败: {e}")
        raise e

# ================ 下面是单独脚本批量处理时的流程 ================

def parse_page_numbers_from_filename(filename):
    """
    根据图片文件名（如 "page_30.png" 或 "page_30-31.png"）解析出页码信息
    """
    name = os.path.splitext(filename)[0]
    if name.startswith("page_"):
        name = name[len("page_"):]
    try:
        return int(name)
    except ValueError:
        return name

def find_temp_images_folder(pdf_folder_path):
    """
    在 pdf_folder_path 下找到名称包含 'temp_pdf_images' 的子文件夹。
    若找到返回该文件夹的绝对路径，否则返回 None。
    如果有多个符合的子文件夹，则取第一个。
    """
    if not os.path.isdir(pdf_folder_path):
        return None
    for subdir in os.listdir(pdf_folder_path):
        full_path = os.path.join(pdf_folder_path, subdir)
        if os.path.isdir(full_path) and ("temp_pdf_images" in subdir.lower()):
            return full_path
    return None

def process_pdf_not_table_descriptions(pdf_info):
    """
    处理单个 PDF: 
      - 忽略包含 'compressed' 的图片
      - OCR 未识别到文本则写 "无文本信息."
      - 整个文件全部成功后，在 JSON 中标记 not_table_describe_done=True
      - 同时更新数据库状态
    """
    pdf_filename = os.path.basename(pdf_info["file_path"])
    pdf_folder_name = os.path.splitext(pdf_filename)[0]

    fund_code = pdf_info.get("fund_code", "")
    fund_folder = None
    for folder in os.listdir(OUTPUT_DIR):
        if folder.startswith(fund_code):
            fund_folder = folder
            break
    if not fund_folder:
        msg = f"未找到基金文件夹，基金代码: {fund_code}，跳过 {pdf_filename}"
        print(msg)
        logger.warning(msg)
        return False

    pdf_folder_path = os.path.join(OUTPUT_DIR, fund_folder, pdf_folder_name)
    if not os.path.exists(pdf_folder_path):
        msg = f"未找到PDF文件夹: {pdf_folder_path}，跳过 {pdf_filename}"
        print(msg)
        logger.warning(msg)
        return False

    # 使用与 LLM 脚本一致的目录查找逻辑
    temp_img_dir = find_temp_images_folder(pdf_folder_path)
    if not temp_img_dir:
        msg = f"未找到包含 'temp_pdf_images' 的子文件夹，跳过 {pdf_filename}"
        print(msg)
        logger.warning(msg)
        return False

    # 与 LLM 脚本统一，固定文件名为 text.json
    json_output_path = os.path.join(pdf_folder_path, "text.json")
    if os.path.exists(json_output_path):
        with open(json_output_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        json_data = {
            "pages": {},
            "metadata": {
                "file_name": pdf_info.get("file_name"),
                "file_path": pdf_info.get("file_path"),
                "date": pdf_info.get("date"),
                "fund_code": pdf_info.get("fund_code"),
                "short_name": pdf_info.get("short_name"),
                "announcement_title": pdf_info.get("announcement_title"),
                "doc_type_1": pdf_info.get("doc_type_1"),
                "doc_type_2": pdf_info.get("doc_type_2"),
                "announcement_link": pdf_info.get("announcement_link"),
                "source_file": pdf_info.get("file_name")
            }
        }

    # 仅处理不含 "compressed" 的图片
    img_files = [
        f for f in os.listdir(temp_img_dir)
        if f.lower().endswith(".png") and "compressed" not in f.lower()
    ]
    print(f"在 {pdf_folder_name} 中找到 {len(img_files)} 个非表格图片（不含 compressed）。")

    all_images_success = True

    for idx, img_file in enumerate(img_files):
        page_num = parse_page_numbers_from_filename(img_file)
        if str(page_num) in json_data["pages"]:
            print(f"页面 {page_num} 已描述，跳过。")
            continue

        image_path = os.path.join(temp_img_dir, img_file)
        print(f"[{idx+1}/{len(img_files)}] 正在处理图片: {img_file}")

        try:
            start_time = time.time()
            ocr_text = None
            try:
                ocr_text = run_paddle_ocr_on_image(image_path)
            except ValueError as ve:
                # 如果是 "PaddleOCR未识别到任何文本", 则视为成功，写 "无文本信息."
                if "PaddleOCR未识别到任何文本" in str(ve):
                    print(f"未识别到任何文本 -> 视为成功: {img_file}")
                    ocr_text = "无文本信息."
                else:
                    raise ve
            elapsed_time = time.time() - start_time
            print(f"OCR处理耗时 {elapsed_time:.2f} 秒。")

            # 写入 JSON
            json_data["pages"][str(page_num)] = {
                "text": ocr_text,
                "metadata": {
                    **pdf_info,
                    "page_num": page_num,
                    "source_file": pdf_info.get("file_name")
                }
            }
            safe_json_dump(json_data, json_output_path)
            print(f"已更新描述文件: {json_output_path}")

        except Exception as e:
            msg = f"文件 {pdf_filename} 中的图片 {img_file} 处理失败: {str(e)}"
            print(msg)
            logger.warning(msg)  # 记录 WARNING
            all_images_success = False

        time.sleep(1)  # 处理间隔

    # 如果所有图片都成功
    if all_images_success:
        json_data["metadata"]["not_table_describe_done"] = True
        safe_json_dump(json_data, json_output_path)
        print(f"已在 {pdf_folder_name}.json 写入 not_table_describe_done=True。")
        
        # 同时更新数据库状态
        try:
            update_pdf_status_in_db(pdf_info.get("file_name"), "not_table_describe_done", "true")
        except Exception as e:
            logger.error(f"更新数据库状态失败，但JSON文件已更新: {e}")

    return all_images_success

def main():
    processed_files = get_pending_files_from_db()
    files_to_process = []
    
    # 遍历数据库查询结果，构建待处理列表
    for fund_code, pdf_list in processed_files.items():
        for pdf_info in pdf_list:
            files_to_process.append(pdf_info)

    if not files_to_process:
        msg = "没有找到需要处理的文件。"
        print(msg)
        logger.info(msg)
        msg_complete = "非表格图片描述全部完成！"
        print(msg_complete)
        logger.info(msg_complete)
        return

    for pdf_info in files_to_process:
        print(f"开始处理 {pdf_info.get('file_name')} 的非表格图片描述...")

        success = process_pdf_not_table_descriptions(pdf_info)
        if success:
            msg = f"已完成 {pdf_info.get('file_name')} 的非表格图片描述。"
            print(msg)
            logger.info(msg)
        else:
            msg = f"{pdf_info.get('file_name')} 处理非表格图片描述失败。"
            print(msg)
            logger.warning(msg)

    msg_final = "非表格图片描述全部完成！"
    print(msg_final)
    logger.info(msg_final)

# 如果脚本被直接执行，则跑 main()
if __name__ == "__main__":
    main()
