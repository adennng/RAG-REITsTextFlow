#矢量页跨页表格检测,多进程处理
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step3_1_detection_vactor_multi_process.py
(矢量页跨页表格检测，多进程)

"""

import os
import fitz
import pdfplumber
from PIL import Image
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from file_paths_config import OUTPUT_DIR, PDF_DIR

import pymysql
from db_config import get_db_announcement_config

# -------------------- 日志设置 --------------------
logger = logging.getLogger("TableDetection")
logger.setLevel(logging.DEBUG)  # 内部记录全部信息

# 创建日志目录，使用相对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'table_processing_vactor_multifile.log')
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(file_formatter)

class FileFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # 保留 WARNING+ 或 特殊提示
        if msg in ["没有找到需要处理的文件。", "矢量表格检测全部完成！状态以更新至数据库中。"]:
            return True
        return record.levelno >= logging.WARNING

file_handler.addFilter(FileFilter())

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)

class TerminalFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # 允许 特殊提示
        if msg in ["没有找到需要处理的文件。", "矢量表格检测全部完成！状态以更新至数据库中。"]:
            return True
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno == logging.INFO and ("处理文件:" in msg or "添加任务:" in msg):
            return True
        return False

console_handler.addFilter(TerminalFilter())
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------- 参数设置 --------------------
THRESHOLD_TOP = 90
THRESHOLD_BOTTOM = 90
DPI = 300

def get_pending_files_from_db():
    """
    从数据库获取需要进行矢量表格检测的文件
    条件: text_extracted='true' AND table_detection_vector_done='false' AND doc_type_1 != '无关'
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
          AND table_detection_vector_done='false' 
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
        logger.error(f"数据库查询失败: {e}")
        raise e

def update_db_table_detection_done(file_info):
    """
    在 announcement.processed_files 中 ON DUPLICATE KEY UPDATE:
      table_detection_vector_done='true'
    若失败 => 抛异常
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
          'false','true','false','false','false','false','false','false','false','false'
        )
        ON DUPLICATE KEY UPDATE
          table_detection_vector_done='true'
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
    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        raise e

def detect_tables_in_page(pdf_path, page_number):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        tables = page.find_tables()
        bboxes = [tbl.bbox for tbl in tables]

        logger.info(f"[Page {page_number + 1}] 表格检测结果: 发现 {len(tables)} 个表格")
        for i, bbox in enumerate(bboxes, 1):
            logger.info(f"  表格{i}边界框: (x0={bbox[0]:.1f}, y0={bbox[1]:.1f}, x1={bbox[2]:.1f}, y1={bbox[3]:.1f})")
        return (len(tables) > 0), bboxes

def convert_pdf_page_to_image(pdf_path, page_number, output_dir):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=DPI)
    img_path = os.path.join(output_dir, f"page_{page_number + 1}.png")
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(img_path, "PNG")
    logger.info(f"  已生成图片: {img_path}")
    doc.close()
    return img_path

def process_pdf(pdf_path, output_dir):
    """
    对单个PDF进行矢量表格检测:
      - 遍历每页, detect_tables_in_page
      - 有表格 => convert_pdf_page_to_image
      - 再检测跨页 => 拼接
    """
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        table_info_list = []
        logger.info("="*50)
        logger.info(f"开始处理PDF(矢量检测): {pdf_path},总页数:{total_pages}")

        for pg_i, page in enumerate(pdf.pages):
            logger.info(f"处理文件: {os.path.basename(pdf_path)} - 第 {pg_i+1}/{total_pages} 页:")
            has_table, bboxes = detect_tables_in_page(pdf_path, pg_i)
            img_path = None
            if has_table:
                img_path = convert_pdf_page_to_image(pdf_path, pg_i, output_dir)
            else:
                logger.info("  本页未检测到表格,跳过生成图片")
            page_height = page.height
            first_bbox = bboxes[0] if has_table else None
            last_bbox = bboxes[-1] if has_table else None

            table_info_list.append({
                "page_num": pg_i,
                "has_table": has_table,
                "first_bbox": first_bbox,
                "last_bbox": last_bbox,
                "page_height": page_height,
                "img_path": img_path
            })

    i = 0
    while i < len(table_info_list) - 1:
        pending_merge = []
        merged_pages = []
        while i < len(table_info_list) - 1:
            current = table_info_list[i]
            nxt = table_info_list[i+1]
            logger.info(f"检查跨页表格: 第{i+1}页 与 第{i+2}页")
            if current["has_table"] and nxt["has_table"]:
                current_last_bbox = current["last_bbox"]
                current_height = current["page_height"]
                next_first_bbox = nxt["first_bbox"]

                if current_last_bbox:
                    dist_current = current_height - current_last_bbox[3]
                else:
                    dist_current = 99999
                if next_first_bbox:
                    dist_next = next_first_bbox[1]
                else:
                    dist_next = 99999

                logger.info(f"  当前页表格底部距离: {dist_current:.1f}点(阈值={THRESHOLD_BOTTOM})")
                logger.info(f"  下页表格顶部距离: {dist_next:.1f}点(阈值={THRESHOLD_TOP})")
                if dist_current <= THRESHOLD_BOTTOM and dist_next <= THRESHOLD_TOP:
                    logger.info("  ✅ 满足跨页条件 => 合并队列")
                    pending_merge.append(current)
                    merged_pages.append(i+1)
                else:
                    break
            else:
                break
            i += 1

        if pending_merge:
            pending_merge.append(table_info_list[i])
            merged_pages.append(i+1)
            logger.info(f"  ⚡ 合并 {merged_pages}页的跨页表格")
            scale = DPI / 72
            merged_imgs = []
            for pginfo in pending_merge:
                if not pginfo["img_path"]:
                    continue
                img = Image.open(pginfo["img_path"])
                if pginfo == pending_merge[0]:
                    crop_lower = int(pginfo["last_bbox"][3] * scale) if pginfo["last_bbox"] else img.height
                    cropped_img = img.crop((0, 0, img.width, crop_lower))
                elif pginfo == pending_merge[-1]:
                    crop_upper = int(pginfo["first_bbox"][1] * scale) if pginfo["first_bbox"] else 0
                    cropped_img = img.crop((0, crop_upper, img.width, img.height))
                else:
                    if pginfo["first_bbox"] and pginfo["last_bbox"]:
                        crop_up = int(pginfo["first_bbox"][1] * scale)
                        crop_lo = int(pginfo["last_bbox"][3] * scale)
                    else:
                        crop_up, crop_lo = 0, img.height
                    cropped_img = img.crop((0, crop_up, img.width, crop_lo))
                merged_imgs.append(cropped_img)

            if merged_imgs:
                total_h = sum(im.height for im in merged_imgs)
                merged_img = Image.new('RGB', (merged_imgs[0].width, total_h))
                y_off = 0
                for mg in merged_imgs:
                    merged_img.paste(mg, (0, y_off))
                    y_off += mg.height
                merged_name = f"page_{'-'.join(map(str, merged_pages))}.png"
                merged_path = os.path.join(output_dir, merged_name)
                merged_img.save(merged_path)
                logger.info(f"  ✅ 已保存合并图片: {merged_path}")
                for pginfo in pending_merge:
                    if pginfo["img_path"] and os.path.exists(pginfo["img_path"]):
                        os.remove(pginfo["img_path"])
                        logger.info(f"  ❌ 删除原始图片: {pginfo['img_path']}")
        i += 1

    logger.info(f"PDF(矢量检测)处理完成: {pdf_path}")
    logger.info("="*50)
    return table_info_list

def process_pdf_file(task):
    """
    执行矢量检测:
      - detect tables
      - 若成功 => update_db_table_detection_done => table_detection_vector_done='true'
      - 若失败 => 不设置
    """
    fund_code, file_info, pdf_path, output_dir = task
    file_name = file_info["file_name"]
    try:
        process_pdf(pdf_path, output_dir)
        # 先写数据库,若失败抛异常
        update_db_table_detection_done(file_info)
        return (fund_code, file_info, None)
    except Exception as e:
        logger.warning(f"处理 {file_name} 失败: {e}")
        return (fund_code, file_info, str(e))

def main():
    try:
        # 从数据库获取待处理文件
        processed_files = get_pending_files_from_db()
    except Exception as e:
        logger.error(f"从数据库获取待处理文件失败: {e}")
        print(f"从数据库获取待处理文件失败: {e}")
        return

    tasks = []
    # 收集待处理
    for fund_code, fund_list in processed_files.items():
        for fi in fund_list:
            fn = fi["file_name"]
            pdf_path = os.path.join(PDF_DIR, fn)
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF不存在: {pdf_path}")
                continue
            pdf_folder_name = os.path.splitext(fn)[0]
            pdf_folder_dir = os.path.join(OUTPUT_DIR, fund_code, pdf_folder_name)
            if not os.path.exists(pdf_folder_dir):
                logger.warning(f"输出子目录不存在: {pdf_folder_dir}")
                continue
            table_img_dir = os.path.join(pdf_folder_dir, "table_image")
            os.makedirs(table_img_dir, exist_ok=True)

            tasks.append((fund_code, fi, pdf_path, table_img_dir))
            logger.info(f"添加任务: {fn}")

    total_count = len(tasks)
    if total_count == 0:
        logger.info("没有找到需要处理的文件。")
        print("没有找到需要处理的文件。")
        return

    success_count = 0
    fail_list = []

    with ProcessPoolExecutor(max_workers=2) as executor:
        future_map = {executor.submit(process_pdf_file, t): t for t in tasks}
        for fut in as_completed(future_map):
            (fund_code, upd_info, err) = fut.result()
            file_name = upd_info["file_name"]
            if err is None:
                success_count += 1
            else:
                fail_list.append((file_name, err))

    remain = total_count - success_count
    print(f"共 {total_count} 个文件需矢量检测, 成功 {success_count} 个, 失败 {remain} 个.")
    if fail_list:
        print("失败文件:")
        for (fname, reason) in fail_list:
            print(f"  - {fname}: {reason}")

    logger.warning(f"待处理文件数:{total_count}, 本次处理:{total_count}, 成功:{success_count}, 剩余:{remain}")
    if fail_list:
        logger.warning(f"失败文件({len(fail_list)})详情:")
        for (fname, reason) in fail_list:
            logger.warning(f"  {fname}, 原因:{reason}")

    # 全部成功 => 提示
    if remain == 0:
        msg = "矢量表格检测全部完成！状态以更新至数据库中。"
        logger.info(msg)
        print(msg)
    else:
        print("矢量检测部分完成,请查看失败详情.")

if __name__ == "__main__":
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()