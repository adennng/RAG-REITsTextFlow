# 表格图片描述第二次处理主脚本，多线程
import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pdf2image import convert_from_path

import pymysql
import db_config

from step4_table_utils_ali_multi_thread import generate_table_description as generate_table_description_ali  # 原有阿里模型
from step4_table_utils_multi_thresd import generate_table_description as generate_table_description_fallback  # 新增fallback模型
from step4_compress_image import compress_image
from step4_describe_not_table_images_PaddleOCR import run_paddle_ocr_on_image  # 最终兜底

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

# =============== 辅助数据库操作 ===============
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

def update_table_describe_db(record):
    """
    将单个图片的描述记录写入数据库 table_describe 表，
    如果记录已存在则更新（依靠 unique_key 由 source_file 与 page_num 生成）。
    """
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
        return True
    except Exception as e:
        msg = f"更新数据库 table_describe 失败: {e}"
        print(msg)
        logging.warning(msg)
        return False
# =============== 结束数据库操作 ===============

def get_pending_files_from_db():
    """
    从数据库获取需要进行表格描述的文件（第二次处理）
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
    可能是单页 '2' 或多页 '2-3-4'。
    """
    name = os.path.splitext(filename)[0]  # 去除扩展名
    if name.startswith("page_"):
        name = name[len("page_"):]
    return name  # 如 "2" 或 "2-3-4"

def describe_image_with_ali(image_path):
    """
    第一步：调用阿里大模型进行描述。成功返回字符串描述，失败抛异常。
    """
    return generate_table_description_ali(image_path)

def describe_image_with_ali_fallback(image_path):
    """
    调用 step4_table_utils_multi_thresd.py 的 generate_table_description 进行描述（fallback）。
    成功返回字符串描述，失败抛异常。
    """
    return generate_table_description_fallback(image_path)

def split_and_describe_pages(pdf_info, multi_page_str, original_image_path, table_img_dir, describe_json_path):
    """
    当阿里模型多次失败后，对合并的图片（如 "page_2-3-4.png"）进行拆分:
      1) 根据 multi_page_str ('2-3-4') 逐页转成单页 PNG，如 page_2.png, page_3.png, page_4.png
      2) 转完后打印 "已将第X页, 第Y页, 第Z页转为图片"
      3) 删除原合并图片
      4) 分别调用 阿里大模型 -> fallback -> PaddleOCR 的顺序依次尝试生成描述
      5) 如果单页也成功，则将描述写入 {pdf_folder_name}_table_describe.json 中，key 为 "page_2.png"、"page_3.png"、"page_4.png" 等
      6) 如果所有拆分出来的页面均成功，则视为原图成功；否则失败
    """
    pdf_path = pdf_info.get("file_path")
    pdf_filename = os.path.basename(pdf_path)

    # 尝试把 '2-3-4' 拆成 [2,3,4]
    try:
        splitted_pages = [int(x) for x in multi_page_str.split('-')]
    except ValueError:
        msg = f"文件 {pdf_filename} 的图片 {original_image_path} 拆页异常，页码非数字格式: {multi_page_str}"
        print(msg)
        logging.warning(msg)
        return False

    # 将pdf对应的这些页转为单页图片
    all_converted_success = True
    converted_pages = []  # 用于保存已成功转图片的页码
    for p in splitted_pages:
        try:
            images = convert_from_path(pdf_path, dpi=300, first_page=p, last_page=p)
            if not images:
                raise ValueError("pdf2image.convert_from_path 返回空列表")
            new_filename = f"page_{p}.png"
            new_path = os.path.join(table_img_dir, new_filename)
            images[0].save(new_path, 'PNG')
            converted_pages.append(p)
        except Exception as e:
            msg = f"文件 {pdf_filename} 的第 {p} 页转图片失败: {e}"
            print(msg)
            logging.warning(msg)
            all_converted_success = False

    if not all_converted_success:
        msg = f"文件 {pdf_filename} 有部分页转换失败。放弃对 {original_image_path} 的拆分描述。"
        print(msg)
        logging.warning(msg)
        return False

    if converted_pages:
        pages_str = "、".join([f"{p}页" for p in converted_pages])
        print(f"已将第{pages_str}转为图片。")

    try:
        os.remove(original_image_path)
        print(f"已删除原合并图片: {original_image_path}")
    except Exception as e:
        msg = f"删除合并图片 {original_image_path} 失败: {e}"
        print(msg)
        logging.warning(msg)

    splitted_all_success = True
    with json_lock:
        if os.path.exists(describe_json_path):
            with open(describe_json_path, 'r', encoding='utf-8') as f:
                table_descriptions = json.load(f)
        else:
            table_descriptions = {}

    for p in splitted_pages:
        single_page_key = f"page_{p}.png"
        if single_page_key in table_descriptions:
            continue

        new_filename = f"page_{p}.png"
        new_path = os.path.join(table_img_dir, new_filename)
        single_page_description = None

        try:
            start_time = time.time()
            single_page_description = generate_table_description_ali(new_path)
            elapsed = time.time() - start_time
            print(f"文件 {pdf_filename} 的第 {p} 页(单页图)阿里大模型描述成功，耗时 {elapsed:.2f} 秒。")
        except Exception as ali_ex:
            msg = f"文件 {pdf_filename} 的第 {p} 页(单页图)用阿里大模型描述失败: {ali_ex}"
            print(msg)
            logging.warning(msg)
            try:
                start_time = time.time()
                single_page_description = describe_image_with_ali_fallback(new_path)
                elapsed = time.time() - start_time
                print(f"文件 {pdf_filename} 的第 {p} 页(单页图)fallback模型描述成功，耗时 {elapsed:.2f} 秒。")
            except Exception as fb_ex:
                msg = f"文件 {pdf_filename} 的第 {p} 页(单页图)fallback也失败: {fb_ex}"
                print(msg)
                logging.warning(msg)
            if single_page_description is None:
                try:
                    start_time = time.time()
                    single_page_description = run_paddle_ocr_on_image(new_path)
                    elapsed = time.time() - start_time
                    print(f"文件 {pdf_filename} 的第 {p} 页(单页图)PaddleOCR描述成功，耗时 {elapsed:.2f} 秒。")
                except Exception as ocr_ex:
                    msg = f"文件 {pdf_filename} 的第 {p} 页(单页图)用PaddleOCR也失败: {ocr_ex}"
                    print(msg)
                    logging.warning(msg)
                    splitted_all_success = False

        if single_page_description:
            record = {
                "page_num": str(p),
                "picture_path": new_path,
                "file_path": pdf_info.get("file_path"),
                "fund_code": pdf_info.get("fund_code"),
                "short_name": pdf_info.get("short_name"),
                "announcement_title": pdf_info.get("announcement_title"),
                "source_file": os.path.basename(pdf_info.get("file_path")),
                "description": single_page_description
            }
            # 先更新数据库
            if not update_table_describe_db(record):
                splitted_all_success = False
            with json_lock:
                table_descriptions[single_page_key] = record
                safe_json_dump(table_descriptions, describe_json_path)
            print(f"文件 {pdf_filename} 的第 {p} 页(单页图)描述已写入 {describe_json_path}")
        else:
            splitted_all_success = False

    return splitted_all_success

def process_single_image(img_file, table_img_dir, pdf_info, describe_json_path):
    """
    并行处理单张图片的核心逻辑：
      1. 首先调用阿里大模型 describe_image_with_ali 生成描述；
      2. 如果失败则调用 compress_image 后再次用阿里模型重试；
      3. 如果仍失败，则调用 fallback 模型；
      4. 若 fallback 也失败，对于单页图则再尝试 PaddleOCR，对于合并图则调用 split_and_describe_pages 拆分单页依次处理；
      5. 成功后先写入数据库 table_describe（采用 INSERT ... ON DUPLICATE KEY UPDATE 方式），再更新 {pdf_folder_name}_table_describe.json 文件；
      6. 返回 True 表示成功，False 表示失败。
    """
    image_path = os.path.join(table_img_dir, img_file)
    pdf_filename = os.path.basename(pdf_info.get("file_path"))
    page_info = parse_page_numbers_from_filename(img_file)  # '2' 或 '2-3-4'

    # 第1步：阿里大模型描述
    try:
        start_time = time.time()
        description = describe_image_with_ali(image_path)
        elapsed_time = time.time() - start_time
        print(f"图片 {img_file} (文件 {pdf_filename}) 生成描述耗时 {elapsed_time:.2f} 秒。")
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
        if not update_table_describe_db(record):
            return False
        with json_lock:
            if os.path.exists(describe_json_path):
                with open(describe_json_path, 'r', encoding='utf-8') as f:
                    table_descriptions = json.load(f)
            else:
                table_descriptions = {}
            table_descriptions[img_file] = record
            safe_json_dump(table_descriptions, describe_json_path)
        print(f"图片 {img_file} 已写入描述文件(阿里成功)。")
        return True
    except Exception as e:
        # 只记录需要的异常信息
        if isinstance(e, ConnectionResetError) or "Read timed out" in str(e):
            # 连接重置或超时错误不记录到日志
            print(f"图片 {img_file} 描述生成失败: 网络连接问题，忽略记录。")
        elif "DashScope API 多次请求未返回有效流式输出" in str(e):
            # DashScope API请求失败错误不记录到日志
            print(f"图片 {img_file} 描述生成失败: DashScope API 请求失败，忽略记录。")
        else:
            # 其他类型的异常依然记录到日志
            msg = f"文件 {pdf_filename} 的图片 {img_file} 描述失败: {e}"
            print(msg)
            logging.warning(msg)

    # 第2步：压缩后再用阿里模型重试
    try:
        compressed_path = compress_image(image_path)
        if compressed_path is None:
            compressed_path = image_path
        print(f"使用压缩图片: {compressed_path} 重新生成描述(阿里)...")
        start_time = time.time()
        description = describe_image_with_ali(compressed_path)
        elapsed_time = time.time() - start_time
        print(f"压缩后图片 {img_file} (文件 {pdf_filename}) 生成描述耗时 {elapsed_time:.2f} 秒。")
        final_image_path = compressed_path
        record = {
            "page_num": page_info,
            "picture_path": final_image_path,
            "file_path": pdf_info.get("file_path"),
            "fund_code": pdf_info.get("fund_code"),
            "short_name": pdf_info.get("short_name"),
            "announcement_title": pdf_info.get("announcement_title"),
            "source_file": os.path.basename(pdf_info.get("file_path")),
            "description": description
        }
        if not update_table_describe_db(record):
            return False
        with json_lock:
            if os.path.exists(describe_json_path):
                with open(describe_json_path, 'r', encoding='utf-8') as f:
                    table_descriptions = json.load(f)
            else:
                table_descriptions = {}
            table_descriptions[img_file] = record
            safe_json_dump(table_descriptions, describe_json_path)
        print(f"图片 {img_file} 已写入描述文件(阿里-压缩后成功)。")
        return True
    except Exception as ex:
        # 只记录需要的异常信息
        if isinstance(ex, ConnectionResetError) or "Read timed out" in str(ex):
            # 连接重置或超时错误不记录到日志
            print(f"压缩后图片 {img_file} 描述生成失败: 网络连接问题，忽略记录。")
        elif "DashScope API 多次请求未返回有效流式输出" in str(ex):
            # DashScope API请求失败错误不记录到日志
            print(f"压缩后图片 {img_file} 描述生成失败: DashScope API 请求失败，忽略记录。")
        else:
            # 其他类型的异常依然记录到日志
            msg = f"文件 {pdf_filename} 的图片 {img_file} (压缩后)阿里描述失败: {ex}"
            print(msg)
            logging.warning(msg)
    # 第3步：调用 fallback 模型
    try:
        fallback_input = compressed_path if 'compressed_path' in locals() and compressed_path else image_path
        start_time = time.time()
        description = describe_image_with_ali_fallback(fallback_input)
        elapsed_time = time.time() - start_time
        print(f"图片 {img_file} (文件 {pdf_filename}) 使用fallback模型描述耗时 {elapsed_time:.2f} 秒。")
        record = {
            "page_num": page_info,
            "picture_path": fallback_input,
            "file_path": pdf_info.get("file_path"),
            "fund_code": pdf_info.get("fund_code"),
            "short_name": pdf_info.get("short_name"),
            "announcement_title": pdf_info.get("announcement_title"),
            "source_file": os.path.basename(pdf_info.get("file_path")),
            "description": description
        }
        if not update_table_describe_db(record):
            return False
        with json_lock:
            if os.path.exists(describe_json_path):
                with open(describe_json_path, 'r', encoding='utf-8') as f:
                    table_descriptions = json.load(f)
            else:
                table_descriptions = {}
            table_descriptions[img_file] = record
            safe_json_dump(table_descriptions, describe_json_path)
        print(f"图片 {img_file} 已写入描述文件(fallback模型成功)。")
        return True
    except Exception as ex2:
        msg = f"文件 {pdf_filename} 的图片 {img_file} fallback描述仍然失败: {ex2}"
        print(msg)
        logging.warning(msg)
        # 对于单页图片（不含 '-'），在 fallback 失败后再尝试 PaddleOCR
        if '-' not in page_info:
            try:
                start_time = time.time()
                description = run_paddle_ocr_on_image(fallback_input)
                elapsed_time = time.time() - start_time
                print(f"图片 {img_file} (文件 {pdf_filename}) 使用PaddleOCR描述耗时 {elapsed_time:.2f} 秒。")
                record = {
                    "page_num": page_info,
                    "picture_path": fallback_input,
                    "file_path": pdf_info.get("file_path"),
                    "fund_code": pdf_info.get("fund_code"),
                    "short_name": pdf_info.get("short_name"),
                    "announcement_title": pdf_info.get("announcement_title"),
                    "source_file": os.path.basename(pdf_info.get("file_path")),
                    "description": description
                }
                if not update_table_describe_db(record):
                    return False
                with json_lock:
                    if os.path.exists(describe_json_path):
                        with open(describe_json_path, 'r', encoding='utf-8') as f:
                            table_descriptions = json.load(f)
                    else:
                        table_descriptions = {}
                    table_descriptions[img_file] = record
                    safe_json_dump(table_descriptions, describe_json_path)
                print(f"图片 {img_file} 已写入描述文件(PaddleOCR成功)。")
                return True
            except Exception as ocr_ex:
                msg = f"文件 {pdf_filename} 的图片 {img_file} 使用PaddleOCR描述失败: {ocr_ex}"
                print(msg)
                logging.warning(msg)
                return False
        else:
            # 若为合并图片，则尝试拆分单页处理
            splitted_success = split_and_describe_pages(
                pdf_info,
                multi_page_str=page_info,
                original_image_path=image_path,
                table_img_dir=table_img_dir,
                describe_json_path=describe_json_path
            )
            if splitted_success:
                print(f"文件 {pdf_filename} 的图片 {img_file} 拆分为单页并成功描述。视为原图成功。")
                return True
            else:
                msg = f"文件 {pdf_filename} 的图片 {img_file} 拆分后依然有页面描述失败。"
                print(msg)
                logging.warning(msg)
                return False

def process_pdf_table_descriptions(pdf_info, max_workers=10):
    """
    并行处理单个 PDF 文件的所有尚未描述的表格图片。
      - 在 pdf_folder_path 下寻找包含 'table_image' 的文件夹
      - 对每张未处理的图片调用 process_single_image
      - 如果任意图片失败，则返回 False，否则返回 True
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

    # 寻找任意包含 'table_image' 的子文件夹
    table_img_dir = None
    for d in os.listdir(pdf_folder_path):
        full_subdir = os.path.join(pdf_folder_path, d)
        if os.path.isdir(full_subdir) and ("table_image" in d.lower()):
            table_img_dir = full_subdir
            break

    if not table_img_dir or not os.path.exists(table_img_dir):
        msg = f"未找到包含 'table_image' 的文件夹，跳过 {pdf_filename}"
        print(msg)
        logging.warning(msg)
        return False

    # 统一使用简短文件名，避免路径过长
    describe_json_path = os.path.join(pdf_folder_path, "table_describe.json")

    if os.path.exists(describe_json_path):
        with open(describe_json_path, 'r', encoding='utf-8') as f:
            table_descriptions = json.load(f)
    else:
        table_descriptions = {}

    img_files = [
        f for f in os.listdir(table_img_dir)
        if f.lower().endswith(".png") and "compressed" not in f.lower()
    ]
    print(f"在 {pdf_folder_name} 中找到 {len(img_files)} 个表格图片 (文件夹: {os.path.basename(table_img_dir)})。")

    to_process = [f for f in img_files if f not in table_descriptions]
    if not to_process:
        print(f"所有图片均已描述，跳过 {pdf_filename}")
        return True
    print(f"有 {len(to_process)} 张图片需要并行处理...")

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
        final_msg = "表格图片描述全部完成！描述信息已写进数据库。"
        print(final_msg)
        logging.warning(final_msg)
        return

    total_files = len(files_to_process)
    print(f"本次需要处理 {total_files} 个文件。")
    logging.warning(f"本次需要处理 {total_files} 个文件。")

    processed_count = 0
    failed_files = []  # 用于记录处理失败的文件及原因

    for idx, pdf_info in enumerate(files_to_process, start=1):
        file_name = pdf_info.get("file_name")
        print(f"\n----- 正在处理第 {idx}/{total_files} 个文件: {file_name} -----")
        success = process_pdf_table_descriptions(pdf_info, max_workers=10)
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
        progress_msg = f"文件层级进度: 已完成 {idx}/{total_files} 个文件"
        print(progress_msg)

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