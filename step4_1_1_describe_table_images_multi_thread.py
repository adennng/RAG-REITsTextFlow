#表格图片描述主脚本——多线程
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pymysql
import db_config

from step4_table_utils_ali_multi_thread import generate_table_description  # 调用生成描述的函数
from step4_compress_image import compress_image  # 调用压缩图片函数
from file_paths_config import OUTPUT_DIR
from common_utils import safe_json_dump

# 获取脚本所在目录，确保日志文件生成在log目录下
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "table_detection.log")

# 创建 FileHandler，并指定 encoding 为 "utf-8"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.addHandler(file_handler)

# 用于并发写 JSON 文件时加锁，避免多线程竞争
json_lock = threading.Lock()

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
    从数据库获取需要进行表格描述的文件
    条件: table_detection_vector_done='true' AND table_detection_scan_done='true' 
          AND table_describe_done='false' AND doc_type_1 != '无关'
    返回: 按fund_code分组的文件列表字典
    """
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = """
            SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
                   doc_type_1, doc_type_2, announcement_link, table_detection_vector_done,
                   table_detection_scan_done, table_describe_done
            FROM processed_files 
            WHERE table_detection_vector_done='true' 
              AND table_detection_scan_done='true' 
              AND table_describe_done='false' 
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

def parse_page_numbers_from_filename(filename):
    """
    根据图片文件名（如 "page_2.png" 或 "page_2-3.png"）解析出页码信息，返回字符串。
    """
    name = os.path.splitext(filename)[0]  # 去除扩展名
    if name.startswith("page_"):
        name = name[len("page_"):]
    return name  # 如 "2" 或 "2-3"

def process_single_image(img_file, table_img_dir, pdf_info, describe_json_path):
    """
    并行处理单张图片的核心逻辑：
      1. 调用 generate_table_description 生成描述，如果失败则尝试压缩后重试
      2. 成功生成描述后，先将信息写入数据库表 table_describe（采用 INSERT ... ON DUPLICATE KEY UPDATE 方式直接更新已有记录）
      3. 数据库写入成功后，再更新 {pdf_folder_name}_table_describe.json 文件
      4. 返回 True/False 表示是否成功（若数据库写入失败则直接返回 False）
    """
    image_path = os.path.join(table_img_dir, img_file)
    try:
        start_time = time.time()
        description = generate_table_description(image_path)
        elapsed_time = time.time() - start_time
        print(f"图片 {img_file} 生成描述耗时 {elapsed_time:.2f} 秒。")
    except Exception as e:
        # 捕获各种失败，但都不要立刻 return，让它继续走到"压缩重试"。
        if isinstance(e, ConnectionResetError):
            print(f"图片 {img_file} 描述生成失败: 远程连接被重置，尝试压缩后重试。")
        elif "Read timed out" in str(e):
            print(f"图片 {img_file} 描述生成失败: 请求超时，尝试压缩后重试。")
        elif "DashScope API 多次请求未返回有效流式输出" in str(e):
            print(f"图片 {img_file} 描述生成失败: DashScope API 请求失败，尝试压缩后重试。")
        else:
            msg = f"图片 {img_file} 描述生成失败: {e}"
            print(msg)
            #logging.warning(msg)
        try:
            # 压缩后重试
            compressed_path = compress_image(image_path)
            if compressed_path is None:
                compressed_path = image_path
            print(f"使用压缩图片: {compressed_path} 重新生成描述...")
            start_time = time.time()
            description = generate_table_description(compressed_path)
            elapsed_time = time.time() - start_time
            print(f"压缩后图片 {img_file} 生成描述耗时 {elapsed_time:.2f} 秒。")
            image_path = compressed_path
        except Exception as ex:
            msg = f"图片 {img_file} 压缩后描述生成失败: {ex}"
            print(msg)
            #logging.warning(msg)
            return False  # 不写 JSON，也不更新数据库

    # 构造记录
    page_info = parse_page_numbers_from_filename(img_file)
    record = {
        "page_num": page_info,
        "picture_path": image_path,
        "file_path": pdf_info.get("file_path"),
        "fund_code": pdf_info.get("fund_code"),
        "short_name": pdf_info.get("short_name"),
        "announcement_title": pdf_info.get("announcement_title"),
        "source_file": os.path.basename(pdf_info.get("file_path")),
        "description": description
    }

    # 先将记录插入或更新至数据库中的 table_describe 表
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO table_describe 
            (fund_code, short_name, announcement_title, source_file, page_num, picture_path, file_path, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                fund_code = VALUES(fund_code),
                short_name = VALUES(short_name),
                announcement_title = VALUES(announcement_title),
                picture_path = VALUES(picture_path),
                file_path = VALUES(file_path),
                description = VALUES(description)
            """
            params = (
                record["fund_code"],
                record["short_name"],
                record["announcement_title"],
                record["source_file"],
                record["page_num"],
                record["picture_path"],
                record["file_path"],
                record["description"]
            )
            cursor.execute(sql, params)
            conn.commit()
        conn.close()
    except Exception as e:
        msg = f"图片 {img_file} 插入数据库失败: {e}"
        print(msg)
        logging.warning(msg)
        return False

    # 数据库写入成功后，立即更新 JSON 文件
    with json_lock:
        if os.path.exists(describe_json_path):
            with open(describe_json_path, 'r', encoding='utf-8') as f:
                table_descriptions = json.load(f)
        else:
            table_descriptions = {}
        table_descriptions[img_file] = record
        safe_json_dump(table_descriptions, describe_json_path)
    print(f"图片 {img_file} 已写入描述文件。")
    return True

def process_pdf_table_descriptions(pdf_info, max_workers=15):
    """
    并行处理单个 PDF 文件的所有尚未描述的表格图片：
      1. 查找该 PDF 文件夹下的 table_image 文件夹
      2. 对每张未处理的图片调用 process_single_image
      3. 如果所有图片均成功，则返回 True；若任意图片处理失败则返回 False
    """
    pdf_filename = os.path.basename(pdf_info["file_path"])
    pdf_folder_name = os.path.splitext(pdf_filename)[0]

    # 根据 fund_code 寻找对应基金文件夹
    fund_code = pdf_info.get("fund_code", "")
    fund_folder = None
    for folder in os.listdir(OUTPUT_DIR):
        if folder.startswith(fund_code):
            fund_folder = folder
            break
    if not fund_folder:
        msg = f"未找到基金文件夹，基金代码: {fund_code}，跳过 {pdf_filename}"
        print(msg)
        logging.warning(msg)
        return False

    pdf_folder_path = os.path.join(OUTPUT_DIR, fund_folder, pdf_folder_name)
    if not os.path.exists(pdf_folder_path):
        msg = f"未找到PDF文件夹: {pdf_folder_path}，跳过 {pdf_filename}"
        print(msg)
        logging.warning(msg)
        return False

    table_img_dir = os.path.join(pdf_folder_path, "table_image")
    if not os.path.exists(table_img_dir):
        msg = f"文件夹 {table_img_dir} 不存在，跳过 {pdf_filename}"
        print(msg)
        logging.warning(msg)
        return False

    # 统一使用简短文件名，避免超长路径/文件名导致报错
    describe_json_path = os.path.join(pdf_folder_path, "table_describe.json")

    # 读取已处理的图片记录
    if os.path.exists(describe_json_path):
        with open(describe_json_path, 'r', encoding='utf-8') as f:
            table_descriptions = json.load(f)
    else:
        table_descriptions = {}

    # 找到所有需要处理的图片
    img_files = [
        f for f in os.listdir(table_img_dir)
        if f.lower().endswith(".png") and "compressed" not in f.lower()
    ]
    print(f"在 {pdf_folder_name} 中找到 {len(img_files)} 个表格图片。")

    # 过滤掉已经描述完成的图片
    to_process = [f for f in img_files if f not in table_descriptions]
    if not to_process:
        print(f"所有图片均已描述，跳过 {pdf_filename}")
        return True
    print(f"有 {len(to_process)} 张图片需要并行处理...")

    # 并行处理
    all_images_success = True
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image, img_file, table_img_dir, pdf_info, describe_json_path
            ): img_file
            for img_file in to_process
        }
        for future in as_completed(futures):
            img_name = futures[future]
            success = future.result()
            if not success:
                all_images_success = False

    if all_images_success:
        print(f"完成 {pdf_folder_name} 的表格描述（并行处理）。")
    else:
        print(f"{pdf_folder_name} 中有部分图片处理失败。")
    return all_images_success

def main():
    try:
        # 从数据库获取待处理文件
        processed_files = get_pending_files_from_db()
    except Exception as e:
        error_msg = f"从数据库获取待处理文件失败: {e}"
        print(error_msg)
        logging.error(error_msg)
        return

    # 收集待处理PDF
    files_to_process = []
    for fund_code, pdf_list in processed_files.items():
        files_to_process.extend(pdf_list)

    if not files_to_process:
        msg = "没有找到需要处理的文件。"
        print(msg)
        logging.warning(msg)
        print("表格图片描述全部完成！描述信息已写进数据库。")
        logging.warning("表格图片描述全部完成！描述信息已写进数据库。")
        return

    total_files = len(files_to_process)
    print(f"本次需要处理 {total_files} 个文件。")
    logging.info(f"待处理文件总数: {total_files}")

    processed_count = 0
    failed_files = []  # 用于记录处理失败的文件及原因

    # 文件层级处理
    for idx, pdf_info in enumerate(files_to_process, start=1):
        file_name = pdf_info.get("file_name")
        print(f"\n----- 正在处理第 {idx}/{total_files} 个文件: {file_name} -----")
        logging.info(f"开始处理文件: {file_name}")
        success = process_pdf_table_descriptions(pdf_info, max_workers=15)
        if success:
            # 只更新数据库 processed_files 表
            try:
                conn = get_announcement_connection()
                with conn.cursor() as cursor:
                    sql = "UPDATE processed_files SET table_describe_done = %s WHERE file_name = %s"
                    cursor.execute(sql, ("true", file_name))
                    conn.commit()
                conn.close()
                msg = f"已更新 {file_name} 状态为 table_describe_done=True。"
                print(msg)
                logging.info(msg)
                processed_count += 1
            except Exception as e:
                msg = f"文件 {file_name} 更新数据库 processed_files 失败: {e}"
                print(msg)
                logging.warning(msg)
                failed_files.append((file_name, f"更新数据库 processed_files 失败: {e}"))
        else:
            msg = f"{file_name} 处理表格描述失败。"
            print(msg)
            logging.warning(msg)
            failed_files.append((file_name, "部分图片处理失败"))
        print(f"文件层级进度: 已完成 {idx}/{total_files} 个文件")

    remaining_files = total_files - processed_count
    summary_msg = (
        f"\n处理总结：\n"
        f"待处理文件总数: {total_files}\n"
        f"本次成功处理: {processed_count}\n"
        f"未处理文件: {remaining_files}\n"
    )
    print(summary_msg)
    logging.warning(summary_msg)

    if failed_files:
        fail_msg = "处理失败的文件及原因：\n" + "\n".join([f"文件: {fn}, 原因: {reason}" for fn, reason in failed_files])
        print(fail_msg)
        logging.warning(fail_msg)

    final_msg = "表格图片描述全部完成！描述信息已写进数据库。"
    print(f"\n{final_msg}")
    logging.warning(final_msg)

if __name__ == "__main__":
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()