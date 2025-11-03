# 非表格图片描述的主脚本——llm版（多线程修改版）
import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import pymysql
import db_config

from step4_table_utils_multi_thresd import generate_table_description  # 调用生成描述的函数
from step4_compress_image import compress_image  # 调用压缩图片函数
from file_paths_config import OUTPUT_DIR
from common_utils import safe_json_dump

# 配置参数
MAX_THREADS = 3  # 最大线程数

# ================= 自定义日志过滤器 =================
# 允许以下情况写入日志文件：
# 1) 级别>=WARNING 的消息；
# 2) 消息内容恰好是「没有找到需要处理的文件。」或「非表格图片描述全部完成！」（即便是INFO级别，也要写入日志）。
class SpecificLogFilter(logging.Filter):
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

# ================ 配置日志 ================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 总体日志级别设为 INFO

file_handler = logging.FileHandler(log_file, encoding='utf-8')
# 注意，这里要设为 INFO 级别，不要设为 WARNING，否则过滤器无法捕捉到我们需要单独放行的 "INFO" 消息。
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# 添加自定义过滤器
file_handler.addFilter(SpecificLogFilter())

# 确保只添加一次handler，避免重复日志
if not logger.handlers:
    logger.addHandler(file_handler)
else:
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    logger.addHandler(file_handler)
# ==========================================

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
    从数据库获取需要进行非表格图片描述的文件
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
    """
    根据图片文件名（如 "page_30.png" 或 "page_30-31.png"）解析出页码信息，返回字符串或整数。
    """
    name = os.path.splitext(filename)[0]  # 去除扩展名
    if name.startswith("page_"):
        name = name[len("page_"):]
    try:
        return int(name)  # 将页码转换为整数返回
    except ValueError:
        return name  # 如果转换失败，则原样返回

def process_pdf_not_table_descriptions(pdf_info):
    """
    针对单个 PDF 文件：
      - 根据 pdf_info 中的 fund_code，在 OUTPUT_DIR 下查找对应基金文件夹（仅用基金代码匹配），
      - 在该基金文件夹下查找以 PDF 文件名（去扩展名）命名的文件夹，
      - 在该 PDF 文件夹下定位子目录 temp_pdf_images，存放了从 PDF 中提取的非表格图片（PNG 格式），
      - 对每张图片调用大模型生成描述。如果生成失败，则先调用压缩图片再重试；
        若压缩后仍异常，则打印并记录错误，不写入该图片信息，直接跳过该图片，
      - 每处理完一张图片，立即写入 JSON 文件，JSON 文件名为 "{PDF文件名}.json"，
        其结构包括 "pages"（按页分组，页数取自图片名称，如 "page_30.png" 得 "30"）和 "metadata"（从 pdf_info 中获取相关信息），
      - 如果某个 PDF 文件中任一图片处理失败，则返回 False（表示该文件整体描述未成功），
      - 返回 True 表示该 PDF 文件所有图片均成功生成描述。
    """
    pdf_filename = os.path.basename(pdf_info["file_path"])
    pdf_folder_name = os.path.splitext(pdf_filename)[0]

    # 根据 fund_code 寻找对应基金文件夹（仅用基金代码匹配）
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

    # 非表格图片所在目录：temp_pdf_images
    temp_img_dir = os.path.join(pdf_folder_path, "temp_pdf_images")
    if not os.path.exists(temp_img_dir):
        msg = f"文件夹 {temp_img_dir} 不存在，跳过 {pdf_filename}"
        print(msg)
        logger.warning(msg)
        return False

    # 遍历 temp_pdf_images 文件夹中的所有 PNG 图片（排除文件名中包含 "compressed" 的图片）
    img_files = [
        f for f in os.listdir(temp_img_dir)
        if f.lower().endswith(".png") and "compressed" not in f.lower()
    ]

    print(f"在 {pdf_folder_name} 中找到 {len(img_files)} 个非表格图片。")

    all_images_success = True  # 标记是否所有图片均成功生成描述

    # 统一 json 文件名与 step2 保持一致
    json_output_path = os.path.join(pdf_folder_path, "text.json")

    for idx, img_file in enumerate(img_files):
        # 获取图片对应的页码
        page_num = parse_page_numbers_from_filename(img_file)
        image_path = os.path.join(temp_img_dir, img_file)
        print(f"[{idx+1}/{len(img_files)}] 正在处理图片: {img_file}")
        # 如果 text.json 已经包含该页描述，则跳过处理
        if os.path.exists(json_output_path):
            try:
                with open(json_output_path, 'r', encoding='utf-8') as preview_f:
                    preview_data = json.load(preview_f)
                    if str(page_num) in preview_data.get("pages", {}):
                        print(f"页面 {page_num} 已描述，跳过。")
                        continue
            except Exception:
                # 如果读取 JSON 失败，则继续处理，后续会重新写入
                pass
        description = None
        try:
            start_time = time.time()
            description = generate_table_description(image_path)
            elapsed_time = time.time() - start_time
            print(f"生成描述耗时 {elapsed_time:.2f} 秒。")
        except Exception as e:
            msg = f"文件 {pdf_filename} 中的图片 {img_file} 描述生成失败: {e}"
            print(msg)
            logger.warning(msg)
            try:
                compressed_path = compress_image(image_path)
                if compressed_path is None:
                    compressed_path = image_path
                print(f"使用压缩图片: {compressed_path} 重新生成描述...")
                start_time = time.time()
                description = generate_table_description(compressed_path)
                elapsed_time = time.time() - start_time
                print(f"压缩后图片生成描述耗时 {elapsed_time:.2f} 秒。")
                image_path = compressed_path
            except Exception as ex:
                msg = f"文件 {pdf_filename} 中的图片 {img_file} 压缩后描述生成失败: {ex}"
                print(msg)
                logger.warning(msg)
                all_images_success = False
                continue

        # ---------------- 追加写入逻辑 ----------------
        # 1) 读取已有 JSON（如不存在则置空）
        if os.path.exists(json_output_path):
            try:
                with open(json_output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}
        else:
            existing_data = {}

        pages_dict = existing_data.get("pages", {})

        page_metadata = {
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

        # 转成字符串 key，保持与 step2 一致
        pages_dict[str(page_num)] = {
            "text": description,
            "metadata": page_metadata
        }

        pdf_metadata = existing_data.get("metadata", {
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
        })

        final_data = {
            "pages": pages_dict,
            "metadata": pdf_metadata
        }

        safe_json_dump(final_data, json_output_path)
        print(f"已更新描述文件: {json_output_path}")

        # 为避免资源浪费，每处理完一张图片休息 1 秒
        time.sleep(1)

    print(f"完成 {pdf_folder_name} 的非表格图片描述。")
    return all_images_success

def process_single_pdf(pdf_info, lock):
    """
    多线程处理单个PDF文件的包装函数。
    在处理成功时只更新数据库 processed_files 表中对应记录的 not_table_describe_done 字段。
    返回 True 表示成功，False 表示失败。
    """
    file_name = pdf_info.get("file_name")
    print(f"开始处理 {file_name} 的非表格图片描述...")
    success = process_pdf_not_table_descriptions(pdf_info)

    if success:
        # 只更新数据库 processed_files 表
        try:
            conn = get_announcement_connection()
            with conn.cursor() as cursor:
                sql = "UPDATE processed_files SET not_table_describe_done = %s WHERE file_name = %s"
                cursor.execute(sql, ("true", file_name))
                conn.commit()
            conn.close()
            msg = f"已更新 {file_name} 状态为 not_table_describe_done=True。"
            print(msg)
            return True
        except Exception as e:
            msg = f"文件 {file_name} 更新数据库 processed_files 失败: {e}"
            print(msg)
            logger.warning(msg)
            return False
    else:
        msg = f"{file_name} 处理非表格图片描述失败。"
        print(msg)
        logger.warning(msg)
        return False

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

    if not files_to_process:
        msg_no_files = "没有找到需要处理的文件。"
        print(msg_no_files)
        logger.info(msg_no_files)  # 此 INFO 信息会被过滤器放行写入日志文件
        msg_complete = "非表格图片描述全部完成！"
        print(msg_complete)
        logger.info(msg_complete)
        return

    total_files = len(files_to_process)
    logger.warning(f"本次需要处理 {total_files} 个文件。")
    print(f"本次需要处理 {total_files} 个文件。")

    success_count = 0
    failed_files = []  # 记录失败文件及原因

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        lock = threading.Lock()
        futures_dict = {}
        for pdf_info in files_to_process:
            future = executor.submit(process_single_pdf, pdf_info, lock)
            futures_dict[future] = pdf_info.get("file_name")
        
        for future in futures_dict:
            file_name = futures_dict[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_files.append((file_name, "非表格图片描述处理失败"))
            except Exception as e:
                failed_files.append((file_name, str(e)))
                logger.error(f"处理 {file_name} 时发生异常: {str(e)}", exc_info=True)

    remaining = total_files - success_count
    summary_msg = (
        f"\n处理总结：\n"
        f"待处理文件总数: {total_files}\n"
        f"本次成功处理: {success_count}\n"
        f"未处理文件: {remaining}\n"
    )
    print(summary_msg)
    logger.warning(summary_msg)
    if failed_files:
        fail_msg = "处理失败的文件及原因：\n" + "\n".join([f"文件: {fn}, 原因: {reason}" for fn, reason in failed_files])
        print(fail_msg)
        logger.warning(fail_msg)

    final_msg = "非表格图片描述全部完成！"
    print(final_msg)
    logger.info(final_msg)  # 此 INFO 消息会被过滤器放行写入日志文件

if __name__ == "__main__":
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()