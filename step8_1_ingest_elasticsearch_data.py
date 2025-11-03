#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 将 mysql 表 text_segmentation_embedding 中信息导入 elasticsearch 里


import os
import time
import json
import logging
import pymysql
from typing import List
from datetime import date as DateObj
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch, helpers

from db_config import get_elasticsearch_config, get_db_announcement_config
import file_paths_config
from common_utils import safe_json_dump

MULTIFILE_OUTPUT_DIR = file_paths_config.OUTPUT_DIR

INDEX_NAME = "reits_announcements"
# 获取脚本所在目录，确保日志文件生成在log目录下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, "es_database.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 可调最大并发线程数
MAX_WORKERS = 5

# 初始化 ES
es_config = get_elasticsearch_config()
scheme = es_config.get('scheme', 'http')  # 默认使用 http
es = Elasticsearch(
    [f"{scheme}://{es_config['host']}:{es_config['port']}"],
    basic_auth=(es_config['username'], es_config['password']),
    verify_certs=False
)

def get_announcement_connection():
    """
    获取数据库 announcement 的连接
    """
    config = get_db_announcement_config()
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
    从数据库获取需要进行ES数据入库的文件
    条件: embedding_done='true' AND elasticsearch_database_done='false' AND doc_type_1 != '无关'
    返回: 按fund_code分组的文件列表字典
    """
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = """
            SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
                   doc_type_1, doc_type_2, announcement_link, embedding_done, elasticsearch_database_done
            FROM processed_files 
            WHERE embedding_done='true' 
              AND elasticsearch_database_done='false' 
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



def fetch_text_chunks_for_file(source_file):
    """
    从数据库 text_segmentation_embedding 表中获取指定 source_file 的文本块，
    返回包含 id, global_id, chunk_id, file_path, date, fund_code, short_name,
    announcement_title, doc_type_1, doc_type_2, announcement_link, source_file, page_num,
    picture_path, char_count, prev_chunks, next_chunks, text 等字段（不取 embedding），
    并按 chunk_id 排序。
    """
    db_config = get_db_announcement_config()
    rows = []
    conn = None
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cursor:
            sql = """
            SELECT
              id,
              global_id,
              chunk_id,
              file_path,
              date,
              fund_code,
              short_name,
              announcement_title,
              doc_type_1,
              doc_type_2,
              announcement_link,
              source_file,
              page_num,
              picture_path,
              char_count,
              prev_chunks,
              next_chunks,
              text
            FROM text_segmentation_embedding
            WHERE source_file = %s
            ORDER BY chunk_id
            """
            cursor.execute(sql, (source_file,))
            rows = cursor.fetchall()
    except Exception as e:
        logger.warning(f"读取数据库失败: {e}")
    finally:
        if conn:
            conn.close()

    result = []
    for row in rows:
        dt = row[4]
        if isinstance(dt, DateObj):
            dt = dt.isoformat()  # 'YYYY-MM-DD'
        else:
            dt = str(dt) if dt else ""
        row_dict = {
            "id": row[0] or 0,
            "global_id": row[1] or "",
            "chunk_id": row[2] or 0,
            "file_path": str(row[3]) if row[3] else "",
            "date": dt,
            "fund_code": str(row[5]) if row[5] else "",
            "short_name": str(row[6]) if row[6] else "",
            "announcement_title": str(row[7]) if row[7] else "",
            "doc_type_1": str(row[8]) if row[8] else "",
            "doc_type_2": str(row[9]) if row[9] else "",
            "announcement_link": str(row[10]) if row[10] else "",
            "source_file": str(row[11]) if row[11] else "",
            "page_num": str(row[12]) if row[12] else "",
            "picture_path": str(row[13]) if row[13] else "",
            "char_count": row[14] if row[14] else 0,
            "prev_chunks": str(row[15]) if row[15] else "",
            "next_chunks": str(row[16]) if row[16] else "",
            "text": str(row[17]) if row[17] else ""
        }
        result.append(row_dict)
    return result

def delete_existing_from_es(source_file):
    """
    先删除 ES 中旧数据：source_file 字段匹配
    """
    try:
        query_body = {
            "query": {
                "term": {
                    "source_file.keyword": source_file
                }
            }
        }
        res = es.delete_by_query(index=INDEX_NAME, body=query_body)
        print(f"已删除 source_file={source_file} 在ES中旧数据, 删除数量: {res['deleted']}.")
    except Exception as e:
        logger.warning(f"删除ES旧数据失败: {e}")

def bulk_insert_es(source_file, docs):
    """
    批量插入 docs（list of dict）到 ES
    """
    if not docs:
        raise ValueError("无文档可插入ES")
    actions = []
    for doc in docs:
        _id = doc["global_id"] if "global_id" in doc and doc["global_id"] else doc["id"]
        action = {
            "_index": INDEX_NAME,
            "_id": _id,
            "_source": doc
        }
        actions.append(action)
    resp = helpers.bulk(es, actions, raise_on_error=False, raise_on_exception=False)
    return resp

def ingest_single_file(pdf_info):
    """
    单个文件处理逻辑(供线程任务调用):
      1) 从 DB 获取文本块记录（按 source_file 查询）
      2) 删除 ES 中旧数据（同一 source_file）
      3) 批量插入新数据到 ES
      4) 若插入过程中出现错误则删除已插入数据，并返回失败
    """
    source_file = pdf_info.get("file_name", "")
    if not source_file:
        return (False, "缺少 file_name")
    chunk_rows = fetch_text_chunks_for_file(source_file)
    if not chunk_rows:
        return (False, "数据库中无文本块记录")
    docs = []
    for row in chunk_rows:
        docs.append(row)
    delete_existing_from_es(source_file)
    try:
        resp = bulk_insert_es(source_file, docs)
    except Exception as e:
        delete_existing_from_es(source_file)
        return (False, f"插入ES出错: {e}")
    return (True, None)

def main():
    # 从数据库获取待处理文件
    try:
        processed_files = get_pending_files_from_db()
    except Exception as e:
        print(f"获取待处理文件失败: {e}")
        logger.warning(f"获取待处理文件失败: {e}")
        return
    
    if not processed_files:
        print("没有找到需要处理的文件。")
        logger.warning("没有找到需要处理的文件。")
        return

    # 收集需要处理的文件
    files_to_process = []
    for fund_code, pdf_list in processed_files.items():
        for pdf_info in pdf_list:
            files_to_process.append(pdf_info)

    total_count = len(files_to_process)
    if total_count == 0:
        print("没有找到需要处理的文件。")
        logger.warning("没有找到需要处理的文件。")
        return

    print(f"本次需要处理 {total_count} 个文件的ES数据入库(多线程={MAX_WORKERS})...")
    success_count = 0
    failed_details = []  # 存 (file_name, reason)

    # 多线程
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {}
        for pdf_info in files_to_process:
            future = executor.submit(ingest_single_file, pdf_info)
            future_to_file[future] = pdf_info

        for fut in as_completed(future_to_file):
            pdf_info = future_to_file[fut]
            file_name = pdf_info.get("file_name", "")
            try:
                success, err_reason = fut.result()
            except Exception as e:
                success = False
                err_reason = str(e)
            if success:
                # 更新数据库状态
                try:
                    update_pdf_status_in_db(file_name, "elasticsearch_database_done", "true")
                except Exception as e:
                    print(f"文件 {file_name} 更新数据库 processed_files 失败: {e}")
                    logger.warning(f"文件 {file_name} 更新数据库 processed_files 失败: {e}")
                    failed_details.append((file_name, f"更新数据库 processed_files 失败: {e}"))
                    continue
                
                success_count += 1
                print(f"文件 {file_name} ES数据入库成功，已更新数据库状态。")
            else:
                if not err_reason:
                    err_reason = "unknown reason"
                print(f"{file_name} 导入ES失败, 原因: {err_reason}")
                failed_details.append((file_name, err_reason))
            time.sleep(1)

    remain = total_count - success_count
    if success_count == total_count:
        print("es数据库入库全部完成！")
        logger.warning("es数据库入库全部完成！")
    else:
        if success_count == 0:
            print("所有文件ES入库均失败或跳过。")
            logger.warning("所有文件ES入库均失败或跳过。")
        else:
            print(f"有 {remain} 个文件未完成ES入库。")
            logger.warning(f"有 {remain} 个文件未完成ES入库。")

    if failed_details:
        for fname, reason in failed_details:
            logger.warning(f"文件 {fname} ES入库失败, 原因: {reason}")
        failed_files_only = [f for (f, _) in failed_details]
        logger.warning(f"下列文件ES入库处理失败: {failed_files_only}")

if __name__ == "__main__":
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()