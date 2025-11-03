#!/usr/bin/env python
# -*- coding: utf-8 -*-
#找到本轮需要处理的pdf,更新至数据库 announcement.processed_files
"""
step1_process_pdfs.py - 仅依赖数据库版本

主要修改点:
1) 从数据库读取已处理文件列表，不再依赖JSON文件进行去重
2) 在解析到 PDF 文件信息后，仅插入/更新数据库 announcement.processed_files (ON DUPLICATE KEY UPDATE)
3) 移除JSON文件操作逻辑，消除双重状态维护问题
4) 在脚本同目录下记录日志文件 process_pdfs.log，日志级别 WARNING+，并在日志写入关键信息（总PDF数、待处理数、处理成功/失败等）
5) 多线程方式处理PDF，处理成功后仅更新数据库状态
6) 分别使用 get_db_config() 查询 reits库公告信息，和 get_db_announcement_config() 写 announcement库 processed_files 表

"""

import os
import re
import time
import logging
import pymysql
from concurrent.futures import ThreadPoolExecutor, as_completed

from db_config import get_db_config, get_db_announcement_config
from file_paths_config import PDF_DIR, OUTPUT_DIR

# ========== 日志配置 ===========
# 创建日志目录，使用相对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, "process_pdfs.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 1. 查询公告类型(在 reits库)
def get_announcement_info_from_mysql(announcement_title, fund_code, date):
    db_conf = get_db_config()  # reits库
    try:
        conn = pymysql.connect(**db_conf)
        cursor = conn.cursor()
        sql = """
        SELECT 公告类型_一级, 公告类型_二级, 公告链接
        FROM 公告信息
        WHERE 公告标题 = %s AND 基金代码 = %s AND 公告日期 = %s
        """
        cursor.execute(sql, (announcement_title, fund_code, date))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            doc_type_1 = row[0]
            doc_type_2 = row[1] if row[1] else ""
            announcement_link = row[2]
            return doc_type_1, doc_type_2, announcement_link
        else:
            return None, None, None
    except Exception as e:
        print(f"数据库查询错误(reits): {e}")
        return None, None, None

# 2. 写入 announcement.processed_files (ON DUPLICATE KEY UPDATE)
def insert_into_announcement_db(entry):
    """
    往 announcement.processed_files 表插入:
      file_name(pk), file_path, date, fund_code, short_name, announcement_title, doc_type_1, doc_type_2, announcement_link
      以及后续处理字段(text_extracted等)初始值都为 'false'
    若主键冲突 => 更新 file_path, date, fund_code, short_name, announcement_title, doc_type_1, doc_type_2, announcement_link
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
          'false','false','false','false','false','false','false','false','false','false'
        )
        ON DUPLICATE KEY UPDATE
          file_path=VALUES(file_path),
          date=VALUES(date),
          fund_code=VALUES(fund_code),
          short_name=VALUES(short_name),
          announcement_title=VALUES(announcement_title),
          doc_type_1=VALUES(doc_type_1),
          doc_type_2=VALUES(doc_type_2),
          announcement_link=VALUES(announcement_link)
        """
        vals = (
            entry["file_name"],
            entry["file_path"],
            entry["date"],
            entry["fund_code"],
            entry["short_name"],
            entry["announcement_title"],
            entry["doc_type_1"],
            entry["doc_type_2"],
            entry["announcement_link"]
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
        # 抛异常给外部 process_pdf 捕获
        raise e

# 单个 PDF 文件多线程处理
def process_pdf(file):
    # 先判断是否 .pdf
    if not file.lower().endswith('.pdf'):
        return None

    # 用正则提取 date, fund_code, short_name, announcement_title
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{6}\.\w{2})_(.*?)_(.+)\.pdf")
    match = pattern.match(file)
    if not match:
        print(f"文件名格式不匹配: {file}")
        return None

    date, fund_code, short_name, announcement_title = match.groups()
    file_path = os.path.join(PDF_DIR, file)
    print(f"解析文件: {file} => 日期:{date},基金代码:{fund_code},基金简称:{short_name},公告:{announcement_title}")

    # 查询 doc_type_1, doc_type_2, link
    doc_type_1, doc_type_2, announcement_link = get_announcement_info_from_mysql(announcement_title, fund_code, date)
    if doc_type_1 is None:
        print(f"未找到公告类型匹配记录: {file}")
        return None

    entry = {
        "file_name": file,
        "file_path": file_path,
        "date": date,
        "fund_code": fund_code,
        "short_name": short_name,
        "announcement_title": announcement_title,
        "doc_type_1": doc_type_1,
        "doc_type_2": doc_type_2,
        "announcement_link": announcement_link
    }
    # 写数据库
    try:
        ok = insert_into_announcement_db(entry)
        if ok:
            return (fund_code, entry)
        else:
            return None
    except Exception as e:
        print(f"插入数据库失败: {file}, 原因:{e}")
        return None

def get_processed_files_from_db():
    """从数据库获取已处理的文件列表"""
    processed_files = set()
    try:
        db_config = get_db_announcement_config()
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            sql = "SELECT file_name FROM processed_files"
            cursor.execute(sql)
            results = cursor.fetchall()
            processed_files = {row[0] for row in results}
        connection.close()
        print(f"从数据库获取到 {len(processed_files)} 个已处理文件")
        return processed_files
    except Exception as e:
        print(f"数据库查询失败: {str(e)}")
        return set()

def process_pdfs():
    # 从数据库获取已处理文件列表
    processed_files_db = get_processed_files_from_db()
    pdf_files = os.listdir(PDF_DIR)
    total_pdf_files = len(pdf_files)

    # 匹配正则
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{6}\.\w{2})_(.*?)_(.+)\.pdf")

    # 过滤已处理（只检查数据库）
    files_to_process = []
    for file in pdf_files:
        if file.lower().endswith('.pdf'):
            m = pattern.match(file)
            if not m:
                print(f"文件名格式不匹配: {file}")
                continue
            # 只检查数据库中是否已存在
            if file in processed_files_db:
                print(f"文件 {file} 已处理, 跳过.")
                continue
            files_to_process.append(file)

    pending_count = len(files_to_process)
    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(process_pdf, f): f for f in files_to_process}
        for future in as_completed(future_map):
            f = future_map[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
                else:
                    errors.append((f, "返回None"))
            except Exception as e:
                errors.append((f, str(e)))

    # 统计结果（不再写JSON文件）
    inserted_count = len(results)
    
    # 从数据库重新统计
    updated_processed_files_db = get_processed_files_from_db()
    total_processed = len(updated_processed_files_db)
    
    # 按基金代码分组统计
    fund_groups = {}
    try:
        db_config = get_db_announcement_config()
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            sql = "SELECT fund_code, COUNT(*) FROM processed_files GROUP BY fund_code"
            cursor.execute(sql)
            fund_groups = dict(cursor.fetchall())
        connection.close()
    except Exception as e:
        print(f"统计分组失败: {e}")
    group_count = len(fund_groups)

    # 终端输出
    print("\n===========================================")
    print(f"总PDF文件数: {total_pdf_files}")
    print(f"待处理文件数量: {pending_count}")
    print(f"已处理基金组数: {group_count}, 总处理文件数: {total_processed}")
    print(f"本次处理文件数: {len(files_to_process)}, 成功入库: {inserted_count}, 失败: {len(errors)}")
    if errors:
        print("失败文件:")
        for (fname, reason) in errors:
            print(f"  - {fname}: {reason}")
    print("===========================================\n")

    # 写日志
    logger.warning(f"总PDF文件数: {total_pdf_files}")
    logger.warning(f"待处理文件数量: {pending_count}")
    logger.warning(f"已处理基金组数: {group_count}, 总处理文件数: {total_processed}")
    logger.warning(f"本次处理文件数: {len(files_to_process)}, 成功写DB: {inserted_count}, 失败: {len(errors)}")
    if errors:
        logger.warning(f"本次处理失败共 {len(errors)} 个:")
        for (fname, reason) in errors:
            logger.warning(f"  文件: {fname}, 原因: {reason}")

if __name__ == "__main__":
    process_pdfs()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()