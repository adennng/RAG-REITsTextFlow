# 非表格图片第二次描述的主脚本——LLM版（多线程修改版）
# - 若文件夹名包含 "temp_pdf_images" 即视为可用图片目录
# - 统计并打印处理文件总数、成功/失败情况

import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import pymysql
import db_config

# ======= LLM生成描述、压缩图片 =======
from step4_table_utils_multi_thresd import generate_table_description  # 调用生成描述的函数
from step4_compress_image import compress_image  # 调用压缩图片函数

# ======= 引入PaddleOCR提供的函数 =======
from step4_describe_not_table_images_PaddleOCR import run_paddle_ocr_on_image

from file_paths_config import OUTPUT_DIR
from common_utils import safe_json_dump

# 配置参数
MAX_THREADS = 1  # 最大线程数

# ================= 自定义日志过滤器 =================
class SpecificLogFilter(logging.Filter):
    """
    允许以下情况写入日志文件：
      1) 级别>=WARNING 的消息；
      2) 消息内容恰好是「没有找到需要处理的文件。」或「非表格图片描述全部完成！」（即便是INFO级别，也要写入日志）。
    """
    def filter(self, record):
        # 过滤掉包含 "OpenAI API 多次请求未返回有效输出" 的日志信息
        if "OpenAI API 多次请求未返回有效输出" in record.getMessage():
            return False
        # 过滤掉包含 "not_table_describe_done=True" 的日志信息
        if "not_table_describe_done=True" in record.getMessage():
            return False
        # 其他符合条件的日志保留
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler.addFilter(SpecificLogFilter())

if not logger.handlers:
    logger.addHandler(file_handler)
else:
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    logger.addHandler(file_handler)
# =================================================

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
    从数据库获取需要进行非表格图片描述的文件（第二次处理）
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

def parse_page_numbers_from_filename(filename):
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

def process_pdf_not_table_descriptions(pdf_info, lock):
    """
    针对单个 PDF 文件的非表格图片描述（多线程安全版）：
      1. 尝试 LLM 描述 -> 失败则压缩后再试 -> 仍失败则调用 PaddleOCR
      2. PaddleOCR 若识别到文本则写入，若未识别到任何文本也视为成功，写 "无文本信息."
      3. 若所有图片都成功处理，则只更新数据库表 processed_files 中对应记录的字段
    """
    pdf_filename = os.path.basename(pdf_info["file_path"])
    pdf_folder_name = os.path.splitext(pdf_filename)[0]
    fund_code = pdf_info.get("fund_code", "")
    file_name = pdf_info.get("file_name")

    # 找基金文件夹
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

    # 查找包含 "temp_pdf_images" 的文件夹
    temp_img_dir = find_temp_images_folder(pdf_folder_path)
    if not temp_img_dir:
        msg = f"未找到包含 'temp_pdf_images' 的子文件夹，跳过 {pdf_filename}"
        print(msg)
        logger.warning(msg)
        return False

    # 统一与 step2 脚本保持一致的文件名
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

    # 找非表格图片（忽略带 "compressed" 的）
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

        description = None

        # =============== 先尝试 LLM ===============
        try:
            start_time = time.time()
            description = generate_table_description(image_path, retries=1)  # 第一次LLM尝试（仅重试1次）
            elapsed_time = time.time() - start_time
            print(f"生成描述耗时 {elapsed_time:.2f} 秒。")
        except Exception as e:
            msg = f"文件 {pdf_filename} 中的图片 {img_file} 描述生成失败: {e}"
            print(msg)
            logger.warning(msg)
            # ============ 失败后，先尝试压缩再 LLM ============
            try:
                compressed_path = compress_image(image_path)
                if compressed_path is None:
                    compressed_path = image_path
                # 从此处开始，后续流程（LLM 及可能的 OCR）均使用 compressed_path
                image_path = compressed_path
                print(f"使用压缩图片: {image_path} 重新生成描述...")
                start_time = time.time()
                description = generate_table_description(image_path, retries=1)  # 压缩后LLM仅重试1次
                elapsed_time = time.time() - start_time
                print(f"压缩后图片生成描述耗时 {elapsed_time:.2f} 秒。")
            except Exception as ex:
                msg = f"文件 {pdf_filename} 中的图片 {img_file} 压缩后描述生成失败: {ex}"
                print(msg)
                logger.warning(msg)
                # ============ 压缩后LLM也失败，则尝试PaddleOCR ============
                try:
                    ocr_text = None
                    try:
                        ocr_text = run_paddle_ocr_on_image(image_path)
                        print("已使用PaddleOCR成功获取图片文本。")
                    except ValueError as ve:
                        if "PaddleOCR未识别到任何文本" in str(ve):
                            print(f"未识别到任何文本 -> 视为成功: {img_file}")
                            ocr_text = "无文本信息."
                        else:
                            raise ve
                    description = ocr_text
                except Exception as ocr_ex:
                    msg = f"文件 {pdf_filename} 中的图片 {img_file} 使用PaddleOCR也失败: {ocr_ex}"
                    print(msg)
                    logger.warning(msg)
                    all_images_success = False
                    continue  # 跳过该图片

        if not description:
            msg = f"文件 {pdf_filename} 中的图片 {img_file} 未能成功生成描述或OCR内容。"
            print(msg)
            logger.warning(msg)
            all_images_success = False
            continue

        # 使用字符串键保持一致
        json_data["pages"][str(page_num)] = {
            "text": description,
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
                "source_file": pdf_info.get("file_name"),
                "page_num": page_num
            }
        }

        safe_json_dump(json_data, json_output_path)
        print(f"已更新描述文件: {json_output_path}")

        time.sleep(1)

    # 如果所有图片都成功处理，则只更新数据库
    if all_images_success:
        try:
            conn = get_announcement_connection()
            with conn.cursor() as cursor:
                sql = "UPDATE processed_files SET not_table_describe_done = %s WHERE file_name = %s"
                cursor.execute(sql, ("true", file_name))
                conn.commit()
            conn.close()
            print(f"已更新数据库 {file_name} 的 not_table_describe_done=True。")
        except Exception as e:
            warn_msg = f"文件 {file_name} 更新数据库 processed_files 失败: {e}"
            print(warn_msg)
            logger.warning(warn_msg)
            return False

    print(f"完成 {pdf_folder_name} 的非表格图片描述。")
    return all_images_success

def process_single_pdf(pdf_info, lock):
    """
    多线程处理单个PDF文件的包装函数
    """
    print(f"开始处理 {pdf_info.get('file_name')} 的非表格图片描述...")
    success = process_pdf_not_table_descriptions(pdf_info, lock)
    return success

def main():
    try:
        # 从数据库获取待处理文件
        processed_files = get_pending_files_from_db()
    except Exception as e:
        error_msg = f"从数据库获取待处理文件失败: {e}"
        print(error_msg)
        logger.error(error_msg)
        return

    # 收集待处理PDF（数据库查询已经包含了所有筛选条件）
    files_to_process = []
    for fund_code, pdf_list in processed_files.items():
        files_to_process.extend(pdf_list)

    num_to_process = len(files_to_process)
    if num_to_process == 0:
        msg_no_files = "没有找到需要处理的文件。"
        print(msg_no_files)
        logger.info(msg_no_files)
        msg_complete = "非表格图片描述全部完成！"
        print(msg_complete)
        logger.info(msg_complete)
        return

    print(f"本次执行需要处理的文件数量: {num_to_process}")
    logger.info(f"本次执行需要处理的文件数量: {num_to_process}")

    results = []
    failed_files = []  # 用于记录处理失败的文件及原因

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        lock = threading.Lock()
        future_to_pdf = {}
        for pdf_info in files_to_process:
            task_info = pdf_info.copy()
            future = executor.submit(process_single_pdf, task_info, lock)
            future_to_pdf[future] = pdf_info

        for future in future_to_pdf:
            pdf_info = future_to_pdf[future]
            file_name = pdf_info.get("file_name")
            try:
                success = future.result()
                results.append(success)
                if success:
                    msg = f"文件 {file_name} 处理成功。"
                    print(msg)
                    logger.info(msg)
                else:
                    msg = f"文件 {file_name} 处理失败。"
                    print(msg)
                    logger.warning(msg)
                    failed_files.append((file_name, "非表格图片描述处理失败"))
            except Exception as e:
                msg = f"处理 {file_name} 时发生异常: {str(e)}"
                print(msg)
                logger.error(msg, exc_info=True)
                results.append(False)
                failed_files.append((file_name, str(e)))

    files_succeeded = sum(1 for r in results if r)
    files_failed = num_to_process - files_succeeded

    summary_msg = f"\n处理总结：\n待处理文件总数: {num_to_process}\n本次成功处理: {files_succeeded}\n未处理文件: {files_failed}\n"
    print(summary_msg)
    logger.warning(summary_msg)
    if failed_files:
        fail_msg = "处理失败的文件及原因：\n" + "\n".join([f"文件: {fn}, 原因: {reason}" for fn, reason in failed_files])
        print(fail_msg)
        logger.warning(fail_msg)

    msg_final = "非表格图片描述全部完成！"
    print(msg_final)
    logger.info(msg_final)

if __name__ == "__main__":
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()