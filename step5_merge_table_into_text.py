# 表格页描述替换至全文文字里
import os
import re
import json
import time
import logging
import pymysql
import db_config
from file_paths_config import OUTPUT_DIR as MULTIFILE_OUTPUT_DIR
from common_utils import safe_json_dump

# ================ 日志配置（工作日志 merge.log，只记录失败信息和 WARNING 及以上级别消息） ================
# 获取脚本所在目录，确保日志文件生成在log目录下
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "merge.log")

# 定义自定义过滤器：允许 WARNING 及以上，或特定消息
class SpecificLogFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if record.getMessage() in ("没有文件需要处理。", "表格图片描述内容与文本内容合并全部完成！描述信息已写进数据库和json文件里，状态已更新至数据库和json文件。"):
            return True
        return False

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler.addFilter(SpecificLogFilter())
# 确保只添加一次 handler
if not logger.handlers:
    logger.addHandler(file_handler)
else:
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    logger.addHandler(file_handler)
# =======================================================================================================

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
    从数据库获取需要合并表格和文本的文件
    条件: text_extracted='true' AND table_describe_done='true' 
          AND not_table_describe_done='true' AND merge_done='false' 
          AND doc_type_1 != '无关'
    返回: 按fund_code分组的文件列表字典
    """
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = """
            SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
                   doc_type_1, doc_type_2, announcement_link, text_extracted,
                   table_describe_done, not_table_describe_done, merge_done
            FROM processed_files 
            WHERE text_extracted='true' 
              AND table_describe_done='true' 
              AND not_table_describe_done='true' 
              AND merge_done='false' 
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



def load_json_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_json_file(data, path):
    safe_json_dump(data, path)

def parse_page_numbers(page_num_str):
    if "-" in page_num_str:
        parts = page_num_str.split("-")
        return [int(p) for p in parts if p.isdigit()]
    else:
        try:
            return [int(page_num_str)]
        except:
            return []

def update_pages_metadata(text_data, default_metadata):
    """对 text_data 中所有页面条目的 metadata 补充缺失的默认项"""
    pages = text_data.get("pages", {})
    for key, page in pages.items():
        if "metadata" not in page or not isinstance(page["metadata"], dict):
            page["metadata"] = {}
        # 确保 picture_path 和 page_num 至少存在
        page["metadata"].setdefault("picture_path", "")
        page["metadata"].setdefault("page_num", key)
        # 对于其他字段，补充 default_metadata 中缺失的部分
        for meta_key, meta_value in default_metadata.items():
            page["metadata"].setdefault(meta_key, meta_value)
    return text_data

def merge_extracted_text_and_description(text_data, table_desc_data, default_metadata):
    pages = text_data.get("pages", {})

    # 对已有页面先更新 metadata（至少保证 picture_path 和 page_num 存在）
    for key, page in pages.items():
        if "metadata" not in page or not isinstance(page["metadata"], dict):
            page["metadata"] = {}
        page["metadata"].setdefault("picture_path", "")
        page["metadata"].setdefault("page_num", key)

    # 分类收集 _table_describe.json 的记录
    single_pages = {}
    multi_pages = {}
    for img_key, record in table_desc_data.items():
        page_num_str = record.get("page_num", "").strip()
        if not page_num_str:
            continue
        if "-" in page_num_str:
            multi_pages[img_key] = record
        else:
            single_pages[img_key] = record

    # 处理多页描述（优先处理）
    for img_key, record in multi_pages.items():
        page_num_str = record["page_num"].strip()
        description = record.get("description", "").strip()
        picture_path = record.get("picture_path", "").strip()
        
        page_nums = parse_page_numbers(page_num_str)
        if not page_nums:
            continue

        # 删除所有涉及的原有单页
        for pn in page_nums:
            pn_str = str(pn)
            if pn_str in pages:
                del pages[pn_str]

        # 新增多页合并页面，以整个页码范围字符串为 key
        meta = default_metadata.copy()
        meta.update({
            "picture_path": picture_path,
            "page_num": page_num_str
        })
        pages[page_num_str] = {"text": description, "metadata": meta}

    # 处理单页描述
    for img_key, record in single_pages.items():
        page_num_str = record["page_num"].strip()
        description = record.get("description", "").strip()
        picture_path = record.get("picture_path", "").strip()
        page_nums = parse_page_numbers(page_num_str)
        if not page_nums:
            continue
        pn = str(page_nums[0])
        if pn in pages:
            pages[pn]["text"] = description
            pages[pn]["metadata"]["picture_path"] = picture_path
        else:
            meta = default_metadata.copy()
            meta.update({
                "picture_path": picture_path,
                "page_num": pn
            })
            pages[pn] = {"text": description, "metadata": meta}

    text_data["pages"] = pages
    return text_data

# ---------------- 新增：将 text.json 导入数据库 page_data 表 ----------------
def update_page_data_db(pdf_info, merged_data):
    """
    将 merged_data（即 {pdf_folder_name}.json 合并后的内容）导入数据库 announcement 的 page_data 表。  
    先根据 source_file（这里取 pdf_info["file_name"]）删除已有的记录，再逐行插入 merged_data["pages"] 中的每个页面。  
    若成功返回 True，否则返回 False。
    """
    source_file = pdf_info.get("file_name")
    try:
        conn = db_config.get_db_announcement_config()
        conn = pymysql.connect(
            host=conn["host"],
            port=conn["port"],
            user=conn["user"],
            password=conn["password"],
            database=conn["database"],
            charset=conn["charset"],
            cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            # 删除已有的记录（source_file匹配）
            delete_sql = "DELETE FROM page_data WHERE source_file = %s"
            cursor.execute(delete_sql, (source_file,))
            
            # 遍历 merged_data["pages"] 插入每个页面数据
            pages = merged_data.get("pages", {})
            insert_sql = """
            INSERT INTO page_data 
            (file_name, file_path, date, fund_code, short_name, announcement_title, doc_type_1, doc_type_2, announcement_link, source_file, page_num, picture_path, text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            for key, page in pages.items():
                metadata = page.get("metadata", {})
                page_num = str(metadata.get("page_num", key))
                picture_path = metadata.get("picture_path", "")
                text_content = page.get("text", "")
                params = (
                    pdf_info.get("file_name", ""),
                    pdf_info.get("file_path", ""),
                    pdf_info.get("date", ""),
                    pdf_info.get("fund_code", ""),
                    pdf_info.get("short_name", ""),
                    pdf_info.get("announcement_title", ""),
                    pdf_info.get("doc_type_1", ""),
                    pdf_info.get("doc_type_2", ""),
                    pdf_info.get("announcement_link", ""),
                    pdf_info.get("file_name", ""),  # source_file字段使用 pdf_info file_name
                    page_num,
                    picture_path,
                    text_content
                )
                cursor.execute(insert_sql, params)
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        err_msg = f"更新数据库 page_data 失败（源文件 {source_file}）：{e}"
        print(err_msg)
        logger.warning(err_msg)
        return False
# -----------------------------------------------------------------------------------

def process_pdf_text_merge(pdf_key, pdf_info):
    pdf_filename = os.path.basename(pdf_info["file_path"])
    pdf_folder_name = os.path.splitext(pdf_filename)[0]
    fund_code = pdf_info.get("fund_code", "")

    # 直接使用 fund_code 作为文件夹名
    fund_folder = os.path.join(MULTIFILE_OUTPUT_DIR, fund_code)
    if not os.path.exists(fund_folder):
        print(f"基金文件夹不存在: {fund_folder}")
        return {"success": False, "error": f"基金文件夹不存在: {fund_folder}"}

    pdf_folder = os.path.join(fund_folder, pdf_folder_name)
    # 统一文件名：文本内容 → text.json ，表格描述 → table_describe.json
    text_json_path = os.path.join(pdf_folder, "text.json")
    describe_json_path = os.path.join(pdf_folder, "table_describe.json")

    # 构建默认 metadata（来源于 pdf_info 数据库记录）
    default_metadata = {
        "file_name": pdf_info.get("file_name", ""),
        "file_path": pdf_info.get("file_path", ""),
        "date": pdf_info.get("date", ""),
        "fund_code": pdf_info.get("fund_code", ""),
        "short_name": pdf_info.get("short_name", ""),
        "announcement_title": pdf_info.get("announcement_title", ""),
        "doc_type_1": pdf_info.get("doc_type_1", ""),
        "doc_type_2": pdf_info.get("doc_type_2", ""),
        "announcement_link": pdf_info.get("announcement_link", ""),
        "source_file": pdf_info.get("file_name", "")
    }

    # 文件存在性判断
    text_exists = os.path.exists(text_json_path)
    desc_exists = os.path.exists(describe_json_path)
    
    if not text_exists and not desc_exists:
        print(f"两个文件都不存在: {pdf_folder_name}")
        return {"success": False, "error": f"两个文件都不存在: {pdf_folder_name}"}

    merge_success = False
    error_message = ""

    # 仅存在文本文件时（table_describe.json 不存在），补充 picture_path 为空并统一补充 metadata
    if text_exists and not desc_exists:
        print(f"仅文本文件存在: {pdf_folder_name}，无需执行合并。")
        text_data = load_json_file(text_json_path)
        if not text_data:
            text_data = {"pages": {}}
        text_data = update_pages_metadata(text_data, default_metadata)
        if "metadata" not in text_data or not isinstance(text_data["metadata"], dict):
            text_data["metadata"] = default_metadata.copy()
        else:
            for key, value in default_metadata.items():
                text_data["metadata"].setdefault(key, value)
        os.makedirs(os.path.dirname(text_json_path), exist_ok=True)
        save_json_file(text_data, text_json_path)
        # 将文本数据写入数据库 page_data 表
        if not update_page_data_db(pdf_info, text_data):
            error_message = "更新数据库 page_data 失败"
        else:
            merge_success = True

    # 仅存在描述文件时，创建新的文本文件并执行合并
    elif not text_exists and desc_exists:
        print(f"仅表格描述文件存在: {pdf_folder_name}，将创建文本文件并执行合并。")
        text_data = {"pages": {}, "metadata": default_metadata.copy()}
        table_desc_data = load_json_file(describe_json_path) or {}
        merged_data = merge_extracted_text_and_description(text_data, table_desc_data, default_metadata)
        merged_data = update_pages_metadata(merged_data, default_metadata)
        os.makedirs(os.path.dirname(text_json_path), exist_ok=True)
        save_json_file(merged_data, text_json_path)
        print(f"已创建并更新文本文件: {text_json_path}")
        # 将合并后的结果写入数据库 page_data 表
        if not update_page_data_db(pdf_info, merged_data):
            error_message = "更新数据库 page_data 失败"
        else:
            merge_success = True

    # 两个文件均存在，正常处理合并
    else:
        text_data = load_json_file(text_json_path) or {"pages": {}}
        if "metadata" not in text_data or not isinstance(text_data["metadata"], dict):
            text_data["metadata"] = default_metadata.copy()
        else:
            for key, value in default_metadata.items():
                text_data["metadata"].setdefault(key, value)
        table_desc_data = load_json_file(describe_json_path) or {}
        
        merged_data = merge_extracted_text_and_description(text_data, table_desc_data, default_metadata)
        merged_data = update_pages_metadata(merged_data, default_metadata)

        os.makedirs(os.path.dirname(text_json_path), exist_ok=True)
        save_json_file(merged_data, text_json_path)
        print(f"已更新文本文件: {text_json_path}")
        # 更新数据库 page_data 表（先删除同一源文件记录，再插入新的记录）
        if not update_page_data_db(pdf_info, merged_data):
            error_message = "更新数据库 page_data 失败"
        else:
            merge_success = True

    # 如果合并成功，返回成功状态，但不删除图片（图片删除将在数据库更新成功后进行）
    if merge_success:
        return {"success": True, "merge_completed": True, "pdf_folder": pdf_folder}
    else:
        return {"success": False, "error": error_message if error_message else "合并处理失败"}

def main():
    processed_files = get_pending_files_from_db()
    files_to_process = []
    
    # 遍历数据库查询结果，构建待处理列表
    for fund_code, pdf_list in processed_files.items():
        for pdf_info in pdf_list:
            files_to_process.append(pdf_info)

    total_files = len(files_to_process)
    if total_files == 0:
        msg_no_files = "没有文件需要处理。"
        print(msg_no_files)
        logger.warning(msg_no_files)
        return

    print(f"本次执行需要处理的文件数量: {total_files}")
    logger.warning(f"本次执行需要处理的文件数量: {total_files}")

    success_count = 0
    failed_files = []  # 记录 (文件名, 失败原因)
    total_deleted_images = 0

    for pdf_info in files_to_process:
        file_name = pdf_info.get("file_name")
        print(f"开始处理 {file_name} 的描述替换...")
        logger.info(f"开始处理 {file_name} 的描述替换...")
        
        result = process_pdf_text_merge(file_name, pdf_info)
        
        if result["success"]:
            # 先更新数据库状态
            try:
                update_pdf_status_in_db(file_name, "merge_done", "true")
                print(f"已更新 {file_name} 数据库状态为 merge_done=True")
                
                # 数据库更新成功后，再删除图片
                if result.get("merge_completed"):
                    pdf_folder = result.get("pdf_folder")
                    print(f"开始删除文件 {file_name} 对应文件夹中的图片...")
                    deleted_count, failed_deletions = delete_images_in_subfolders(pdf_folder, file_name)
                    
                    if failed_deletions:
                        # 图片删除失败，记录警告但不影响整体成功状态（因为数据库已更新）
                        warn_msg = f"文件 {file_name} 图片删除部分失败: 成功删除 {deleted_count} 个，失败 {len(failed_deletions)} 个"
                        print(warn_msg)
                        logger.warning(warn_msg)
                        for failure in failed_deletions:
                            print(f"  删除失败: {failure}")
                        total_deleted_images += deleted_count
                    else:
                        total_deleted_images += deleted_count
                        
                    succ_msg = f"已完成 {file_name} 的合并和图片清理，删除了 {deleted_count} 个图片文件。"
                else:
                    succ_msg = f"已完成 {file_name} 的合并处理。"
                    
                print(succ_msg)
                logger.info(succ_msg)
                success_count += 1
                
            except Exception as e:
                err_msg = f"文件 {file_name} 更新数据库 processed_files 失败: {e}"
                print(err_msg)
                logger.warning(err_msg)
                failed_files.append((file_name, f"更新数据库状态失败: {e}"))
        else:
            fail_msg = f"{file_name} 合并处理失败: {result.get('error', '未知错误')}"
            print(fail_msg)
            logger.warning(fail_msg)
            failed_files.append((file_name, result.get('error', '未知错误')))

    remaining = total_files - success_count
    summary_msg = (
        f"\n处理总结：\n"
        f"待处理文件总数: {total_files}\n"
        f"本次成功处理: {success_count}\n"
        f"失败文件数量: {len(failed_files)}\n"
        f"总计删除图片: {total_deleted_images} 个\n"
    )
    print(summary_msg)
    logger.warning(summary_msg)
    
    if failed_files:
        fail_details = "处理失败的文件及原因：\n" + "\n".join([f"文件: {fn}, 原因: {reason}" for fn, reason in failed_files])
        print(fail_details)
        logger.warning(fail_details)
        
    if remaining == 0 and total_files > 0:
        final_msg = f"表格图片描述内容与文本内容合并全部完成！描述信息已写进数据库和json文件里，状态已更新至数据库。总计清理图片 {total_deleted_images} 个。"
        print(final_msg)
        logger.warning(final_msg)

def delete_images_in_subfolders(pdf_folder_path, file_name):
    """
    删除指定文件夹下所有子文件夹中的图片文件
    返回: (成功删除的图片数量, 删除失败的图片列表)
    """
    deleted_count = 0
    failed_deletions = []
    
    if not os.path.exists(pdf_folder_path):
        print(f"文件夹不存在: {pdf_folder_path}")
        return 0, [f"文件夹不存在: {pdf_folder_path}"]
    
    try:
        # 遍历所有子文件夹
        for root, dirs, files in os.walk(pdf_folder_path):
            for file in files:
                # 检查是否为图片文件
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg')):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"已删除图片: {file_path}")
                    except Exception as e:
                        failed_deletions.append(f"{file_path}: {str(e)}")
                        print(f"删除图片失败 {file_path}: {str(e)}")
        
        print(f"文件 {file_name} 图片删除完成: 成功删除 {deleted_count} 个图片")
        if failed_deletions:
            print(f"文件 {file_name} 删除失败的图片: {len(failed_deletions)} 个")
            for failure in failed_deletions:
                print(f"  失败: {failure}")
        
        return deleted_count, failed_deletions
        
    except Exception as e:
        error_msg = f"遍历文件夹失败 {pdf_folder_path}: {str(e)}"
        print(error_msg)
        return 0, [error_msg]

if __name__ == "__main__":
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()