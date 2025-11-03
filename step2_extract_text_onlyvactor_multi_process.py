#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step2_a_extract_text_onlyvactor_multi_process.py
文本提取，多进程处理

"""

import os
import regex as re
import json
import datetime
import pdfplumber
import fitz
import numpy as np
from PIL import Image
import concurrent.futures  # 多进程
import logging
import pymysql

from file_paths_config import OUTPUT_DIR, PDF_DIR
from db_config import get_db_announcement_config

PAGE_CHAR_THRESHOLD = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 自定义JSON编码器 ===========
class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理日期时间序列化"""
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)

# ========== 日志配置 ===========
# 创建日志目录，使用相对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, "extract_text_onlyvactor.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
fh = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
fh.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def get_pending_files_from_db():
    """
    从数据库读取需要进行文本提取的文件列表
    返回: list of file_info dict
    """
    db_conf = get_db_announcement_config()
    conn = None
    try:
        conn = pymysql.connect(**db_conf)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 查询text_extracted为false且doc_type_1不为'无关'的记录
        sql = """
        SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
               doc_type_1, doc_type_2, announcement_link
        FROM processed_files 
        WHERE text_extracted = 'false' 
        AND (doc_type_1 != '无关' OR doc_type_1 IS NULL)
        """
        cursor.execute(sql)
        results = cursor.fetchall()
        
        pending_files = []
        for row in results:
            file_info = {
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "date": row["date"],
                "fund_code": row["fund_code"],
                "short_name": row["short_name"],
                "announcement_title": row["announcement_title"],
                "doc_type_1": row["doc_type_1"],
                "doc_type_2": row["doc_type_2"],
                "announcement_link": row["announcement_link"]
            }
            pending_files.append(file_info)
        
        cursor.close()
        conn.close()
        return pending_files
    except Exception as e:
        if conn:
            conn.close()
        raise Exception(f"从数据库读取待处理文件失败: {e}")


def clean_and_reorganize_text(text: str, title_max_length: int = 30) -> str:
    """
    清洗并重新组织矢量提取文本，使其更符合原文排版。
    """
    lines = text.split('\n')
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if len(line) < title_max_length and not re.search(r'\p{P}+$', line):
            cleaned_lines.append(line)
            i += 1
            continue

        current_line = line
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line:
                i += 1
                continue
            # 如果当前行不以句号结尾，并且下一行没有以空格/制表符开始，则合并到同一行
            if not re.search(r'[。]$', current_line) and not next_line.startswith((' ', '\t')):
                current_line += next_line
                i += 1
            else:
                break
        cleaned_lines.append(current_line)
        i += 1

    return '\n'.join(cleaned_lines)


def get_cropped_bbox(pdf_page, top_ratio=0.08, bottom_ratio=0.08):
    """
    返回裁剪掉页眉和页脚的区域，用于矢量文字提取。
    """
    parent_bbox = pdf_page.bbox
    px0, py0, px1, py1 = parent_bbox
    width = px1 - px0
    height = py1 - py0
    top = py0 + height * top_ratio
    bottom = py0 + height * (1 - bottom_ratio)
    return (px0, top, px1, bottom)


def extract_text_from_vector_page(pdf_page) -> str:
    """
    提取矢量页中的文字，并进行简单清洗。
    """
    bbox = get_cropped_bbox(pdf_page)
    cropped_page = pdf_page.within_bbox(bbox)
    text = cropped_page.extract_text() or ""
    return clean_and_reorganize_text(text)


def is_header_or_footer(text: str) -> bool:
    """
    判断文本是否仅是页眉/页脚（如只有页码等）。
    """
    return re.match(r'^\d+$', text.strip()) is not None


def convert_scanned_page_to_image(pdf_path: str, page_number: int, dpi: int, temp_img_dir: str) -> None:
    """
    扫描页只需转为图片存储到 temp_img_dir 文件夹。
    使用 dpi=300，以加快处理并减少磁盘占用。
    在转换前会先检查目标图片是否已存在，若存在则跳过转换。
    """
    image_path = os.path.join(temp_img_dir, f"page_{page_number}.png")
    if os.path.exists(image_path):
        print(f"[转换图片] 第 {page_number} 页图片已存在，跳过转换。")
        return
    print(f"[转换图片] 开始处理PDF第 {page_number} 页...")
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)
    pix = page.get_pixmap(dpi=dpi)
    pix.save(image_path)
    pdf_document.close()
    print(f"[转换图片] 已保存图片至 {image_path}")


def update_announcement_db_text_extracted(file_info) -> bool:
    """
    更新数据库 announcement.processed_files: text_extracted='true'
    若 pk冲突(相同 file_name)，则更新
    若失败抛异常 => 由调用者决定
    """
    db_conf = get_db_announcement_config()
    conn = None
    try:
        conn = pymysql.connect(**db_conf)
        cursor = conn.cursor()
        sql = """
        INSERT INTO processed_files (
            file_name, file_path, date, fund_code, short_name, announcement_title,
            doc_type_1, doc_type_2, announcement_link,
            text_extracted, table_detection_vector_done, table_detection_scan_done,
            table_describe_done, not_table_describe_done, merge_done,
            text_segmentation, embedding_done, vector_database_done, elasticsearch_database_done
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s,
            'true','false','false','false','false','false','false','false','false','false'
        )
        ON DUPLICATE KEY UPDATE
            text_extracted='true'
        """
        vals = (
            file_info["file_name"],
            file_info["file_path"],
            file_info["date"],
            file_info["fund_code"],
            file_info["short_name"],
            file_info["announcement_title"],
            file_info["doc_type_1"],
            file_info["doc_type_2"],
            file_info["announcement_link"]
        )
        cursor.execute(sql, vals)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception:
        if conn:
            conn.rollback()
            conn.close()
        raise


def process_single_file(args):
    """
    子进程处理函数：
    负责处理传入的单个 file_info 所对应的 PDF。

    注意：
    - 每个进程只处理一个 PDF，因此可以安全地进行 PDF 与相应 JSON 的读写。
    - 处理结束后返回更新过的 file_info，用于标记 text_extracted=True。
    - 如果遇到异常则会写入日志并跳过该文件。
    """
    file_info, log_file_name = args
    pdf_path = os.path.join(PDF_DIR, file_info["file_name"])
    file_name = file_info["file_name"]

    # 输出目录
    fund_folder_dir = os.path.join(OUTPUT_DIR, file_info["fund_code"])
    os.makedirs(fund_folder_dir, exist_ok=True)
    pdf_folder_name = os.path.splitext(file_name)[0]
    pdf_folder_dir = os.path.join(fund_folder_dir, pdf_folder_name)
    os.makedirs(pdf_folder_dir, exist_ok=True)
    # 统一使用简短文件名，避免超长路径/文件名导致报错
    output_json_file = os.path.join(pdf_folder_dir, "text.json")

    if os.path.exists(output_json_file):
        with open(output_json_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        file_pages_dict = existing_data.get("pages", {})
    else:
        file_pages_dict = {}
    # 用于存放扫描页转换后的图片
    temp_img_dir = os.path.join(pdf_folder_dir, "temp_pdf_images")
    os.makedirs(temp_img_dir, exist_ok=True)
    # 表格图片目录（如果有）
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"[PDF处理] 文件 {pdf_path} 共 {total_pages} 页")
            for i in range(total_pages):
                page_number = i + 1
                if str(page_number) in file_pages_dict:
                    print(f"[页面跳过] 第 {page_number} 页已处理，跳过。")
                    continue
                print(f"[页面处理] 开始处理第 {page_number} 页")
                page = pdf.pages[i]

                # 矢量文本提取
                vector_text = extract_text_from_vector_page(page)

                # —— 新增：乱码检测 —— #
                # 如果包含 (cid:数字)，视为乱码
                is_cid_garbled = bool(re.search(r'\(cid:\d+\)', vector_text))
                # 计算汉字数量
                han_count = len(re.findall(r'[\u4e00-\u9fff]', vector_text))
                is_low_chinese = han_count < PAGE_CHAR_THRESHOLD

                # 决策：既不是乱码也有足够汉字，才当做矢量页
                if (not is_cid_garbled
                        and not is_low_chinese
                        and len(vector_text) >= PAGE_CHAR_THRESHOLD):

                    current_text = vector_text
                    print(f"[矢量提取成功] 第 {page_number} 页, 文本长度: {len(vector_text)}, 汉字数: {han_count}")

                    if is_header_or_footer(current_text.strip()):
                        print(f"[忽略页眉页脚] 第 {page_number} 页")
                        current_text = ""

                    page_metadata = file_info.copy()
                    page_metadata.update({
                        "source_file": file_info["file_name"],
                        "page_num": page_number
                    })
                    
                    file_pages_dict[str(page_number)] = {
                        "text": current_text,
                        "metadata": page_metadata
                    }

                    final_data = {"pages": file_pages_dict, "metadata": file_info}
                    with open(output_json_file, 'w', encoding='utf-8') as f:
                        json.dump(final_data, f, ensure_ascii=False, indent=4, cls=DateTimeEncoder)
                    print(f"[页面保存] 第 {page_number} 页内容已保存 -> {output_json_file}")

                else:
                    # 当作扫描页：包括文字提取不足或乱码情况
                    print(f"[矢量提取不足或乱码] 第 {page_number} 页, 长度 {len(vector_text)}, 汉字数 {han_count}")
                    convert_scanned_page_to_image(pdf_path, page_number, 300, temp_img_dir)

    except Exception as e:
        print(f"[错误] 文件 {pdf_path} 处理失败: {e}")
        with open(log_file_name, "a", encoding="utf-8") as log_f:
            log_f.write(f"文件 {pdf_path} 处理失败\n原因: {e}\n\n")
        return file_info  # 不改 text_extracted

    # 更新数据库状态
    try:
        update_announcement_db_text_extracted(file_info)
    except Exception as e:
        print(f"[错误] 数据库更新 text_extracted 失败: {file_name}, 原因: {e}")
        return file_info

    file_info["text_extracted"] = True
    return file_info


def main():
    log_file_name = os.path.join(SCRIPT_DIR, "extract_text_onlyvactor_log.txt")

    # 从数据库读取待处理文件
    try:
        pending_files = get_pending_files_from_db()
    except Exception as e:
        print(f"[错误] 从数据库读取待处理文件失败: {e}")
        logger.warning(f"从数据库读取待处理文件失败: {e}")
        return

    tasks = []
    for file_info in pending_files:
        tasks.append((file_info, log_file_name))

    total_to_process = len(tasks)
    if total_to_process == 0:
        print("没有文件需要处理。")
        logger.warning("没有文件需要处理。")
        return

    success_count = 0
    fail_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(process_single_file, t): t for t in tasks}
        for future in concurrent.futures.as_completed(future_map):
            file_info, _ = future_map[future]
            try:
                updated_info = future.result()
                if updated_info.get("text_extracted") is True:
                    success_count += 1
                else:
                    fail_list.append((file_info["file_name"], "未设置 text_extracted"))
            except Exception as e:
                fail_list.append((file_info["file_name"], str(e)))

    remain = total_to_process - success_count
    print(f"[完成] 所有文件处理结束, 成功: {success_count}, 失败: {remain}, 数据库表processed_files已更新.")
    logger.warning(f"[完成] 所有文件处理结束，数据库表processed_files已更新。")
    logger.warning(f"本次处理文件数: {total_to_process}, 成功: {success_count}, 失败: {remain}")
    if fail_list:
        logger.warning("失败列表:")
        for fname, reason in fail_list:
            logger.warning(f"  文件: {fname}, 原因: {reason}")


if __name__ == "__main__":
    main()

# 强制刷新日志
import logging
logging.shutdown()
