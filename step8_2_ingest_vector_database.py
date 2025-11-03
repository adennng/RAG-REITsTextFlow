#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 将 mysql 数据库表 text_segmentation_embedding 中信息导入至向量数据库
"""
主要改动:
1) 使用ThreadPoolExecutor并发处理文件,可调MAX_WORKERS
2) 每个文件流程: 
   - 从DB获取所有字段(含embedding)
   - collection.load()后delete,再insert; 若出错则delete回滚
3) 每个文件成功后更新数据库中 processed_files 表的 vector_database_done 字段为 "true"
4) 在 vector_database.log 中记录WARNING级信息和提示
5) date若为 datetime.date => 转为字符串写入；embedding 不写入ES，只用于关键词检索
"""

import os
import time
import json
import logging
import pymysql
from typing import List
from datetime import date as DateObj
from concurrent.futures import ThreadPoolExecutor, as_completed

from db_config import get_db_announcement_config, get_vector_db_config
import file_paths_config
from common_utils import safe_json_dump
from pymilvus import connections, Collection, utility

MULTIFILE_OUTPUT_DIR = file_paths_config.OUTPUT_DIR

COLLECTION_NAME = "reits_announcement"
# 获取脚本所在目录，确保日志文件生成在log目录下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, "vector_database.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 可调最大并发线程数
MAX_WORKERS = 5

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
    从数据库获取需要进行向量数据库入库的文件
    条件: embedding_done='true' AND vector_database_done='false' AND doc_type_1 != '无关'
    返回: 按fund_code分组的文件列表字典
    """
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = """
            SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
                   doc_type_1, doc_type_2, announcement_link, embedding_done, vector_database_done
            FROM processed_files 
            WHERE embedding_done='true' 
              AND vector_database_done='false' 
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
    查询数据库 text_segmentation_embedding 获取19列:
    (id, global_id, chunk_id, file_path, date, fund_code, short_name, announcement_title, doc_type_1, doc_type_2, 
     announcement_link, source_file, page_num, picture_path, char_count, prev_chunks, next_chunks, text, embedding)
    返回 list of tuple, 并将 date/datetime 转为字符串, embedding 转为 list[float].
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
                text,
                embedding
            FROM text_segmentation_embedding
            WHERE source_file=%s
            ORDER BY chunk_id
            """
            cursor.execute(sql, (source_file,))
            rows = cursor.fetchall()
    except Exception as e:
        logger.warning(f"读取数据库 text_segmentation_embedding 失败: {e}")
    finally:
        if conn:
            conn.close()

    result = []
    import json
    for row in rows:
        db_id = row[0] if row[0] else 0
        global_id = row[1] or ""
        chunk_id = row[2] if row[2] else 0
        file_path = str(row[3]) if row[3] else ""
        dt = row[4]
        if isinstance(dt, DateObj):
            dt = dt.isoformat()
        else:
            dt = str(dt) if dt else ""
        fund_code = str(row[5]) if row[5] else ""
        short_name = str(row[6]) if row[6] else ""
        ann_title = str(row[7]) if row[7] else ""
        doc1 = str(row[8]) if row[8] else ""
        doc2 = str(row[9]) if row[9] else ""
        ann_link = str(row[10]) if row[10] else ""
        s_file = str(row[11]) if row[11] else ""
        page_num = str(row[12]) if row[12] else ""
        pic_path = str(row[13]) if row[13] else ""
        ch_count = row[14] if row[14] else 0
        prev_c = str(row[15]) if row[15] else ""
        next_c = str(row[16]) if row[16] else ""
        txt = str(row[17]) if row[17] else ""
        emb_str = row[18] or ""
        try:
            embedding = json.loads(emb_str) if emb_str else []
        except:
            embedding = []
        result.append((
            db_id, global_id, chunk_id, file_path, dt, fund_code, short_name,
            ann_title, doc1, doc2, ann_link, s_file, page_num, pic_path,
            ch_count, prev_c, next_c, txt, embedding
        ))
    return result

def delete_existing_in_milvus(collection, source_file):
    """
    先 delete expr="source_file=='xxx'", 需 collection.load()后再删
    """
    expr = f"source_file == '{source_file}'"
    try:
        collection.load()
        collection.delete(expr=expr)
        print(f"已删除 source_file={source_file} 在 Milvus 中旧数据。")
    except Exception as e:
        logger.warning(f"删除 Milvus 旧数据失败: {e}")

def insert_to_milvus(collection, source_file, chunk_rows):
    """
    根据 schema 将数据插入集合 COLLECTION_NAME
    支持分批插入大文件数据，避免 gRPC 消息大小超限
    """
    if not chunk_rows:
        raise ValueError("该文件数据库中无文本块记录或embedding为空")
    
    collection.load()
    
    # 分批插入，每批最多 100 条记录
    batch_size = 100
    total_inserted = 0
    
    for i in range(0, len(chunk_rows), batch_size):
        batch_rows = chunk_rows[i:i+batch_size]
        
        ids = []
        global_ids = []
        chunk_ids = []
        file_paths = []
        dates = []
        fund_codes = []
        short_names = []
        ann_titles = []
        doc1s = []
        doc2s = []
        ann_links = []
        source_files = []
        page_nums = []
        pic_paths = []
        char_counts = []
        prev_chs = []
        next_chs = []
        texts = []
        embeddings = []
        
        for row in batch_rows:
            ids.append(row[0])
            global_ids.append(row[1])
            chunk_ids.append(row[2])
            file_paths.append(row[3])
            dates.append(row[4])
            fund_codes.append(row[5])
            short_names.append(row[6])
            ann_titles.append(row[7])
            doc1s.append(row[8])
            doc2s.append(row[9])
            ann_links.append(row[10])
            source_files.append(row[11])
            page_nums.append(row[12])
            pic_paths.append(row[13])
            char_counts.append(row[14])
            prev_chs.append(row[15])
            next_chs.append(row[16])
            texts.append(row[17])
            embeddings.append(row[18])
        
        data = [
            ids, global_ids, chunk_ids, file_paths, dates, fund_codes, short_names,
            ann_titles, doc1s, doc2s, ann_links, source_files, page_nums,
            pic_paths, char_counts, prev_chs, next_chs, texts, embeddings
        ]
        
        try:
            resp = collection.insert(data)
            total_inserted += len(batch_rows)
            print(f"成功插入批次 {i//batch_size + 1}: {len(batch_rows)} 条数据到集合 '{COLLECTION_NAME}'。")
        except Exception as e:
            # 如果批次插入失败，尝试进一步减小批次大小
            if "message larger than max" in str(e) and len(batch_rows) > 10:
                print(f"批次太大，尝试更小的批次大小...")
                smaller_batch_size = 10
                for j in range(0, len(batch_rows), smaller_batch_size):
                    small_batch = batch_rows[j:j+smaller_batch_size]
                    small_data = [
                        [row[k] for row in small_batch] for k in range(19)
                    ]
                    try:
                        resp = collection.insert(small_data)
                        total_inserted += len(small_batch)
                        print(f"成功插入小批次: {len(small_batch)} 条数据。")
                    except Exception as inner_e:
                        print(f"小批次插入也失败: {inner_e}")
                        raise inner_e
            else:
                raise e
    
    print(f"文件 {source_file} 总共成功插入 {total_inserted} 条数据到集合 '{COLLECTION_NAME}'。")

def ingest_file_to_milvus(pdf_info):
    """
    单个文件处理逻辑(线程任务):
      1) 从DB获取文本块记录（按 source_file 查询）
      2) 打开 Milvus Collection
      3) 删除旧数据
      4) 插入新数据
      若异常则删除已插入数据以模拟回滚, 返回 (True, None) 或 (False, error_reason)
    """
    source_file = pdf_info.get("file_name", "")
    if not source_file:
        return (False, "缺少 file_name")
    chunk_rows = fetch_text_chunks_for_file(source_file)
    if not chunk_rows:
        return (False, "数据库中无文本块或embedding信息")
    try:
        collection = Collection(name=COLLECTION_NAME)
    except Exception as e:
        return (False, f"打开 Milvus Collection失败: {e}")
    delete_existing_in_milvus(collection, source_file)
    try:
        insert_to_milvus(collection, source_file, chunk_rows)
    except Exception as e:
        delete_existing_in_milvus(collection, source_file)
        return (False, f"插入 Milvus 出错: {e}")
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

    print(f"本次需要处理 {total_count} 个文件的向量数据库入库(多线程={MAX_WORKERS})...")
    success_count = 0
    failed_details = []  # 存 (file_name, reason)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {}
        for pdf_info in files_to_process:
            future = executor.submit(ingest_file_to_milvus, pdf_info)
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
                    update_pdf_status_in_db(file_name, "vector_database_done", "true")
                except Exception as e:
                    print(f"文件 {file_name} 更新数据库 processed_files 失败: {e}")
                    logger.warning(f"文件 {file_name} 更新数据库 processed_files 失败: {e}")
                    failed_details.append((file_name, f"更新数据库 processed_files 失败: {e}"))
                    continue
                
                success_count += 1
                print(f"文件 {file_name} 向量数据库入库成功，已更新数据库状态。")
            else:
                if not err_reason:
                    err_reason = "unknown reason"
                print(f"{file_name} 向量数据库入库失败，原因: {err_reason}")
                failed_details.append((file_name, err_reason))
            time.sleep(1)

    remain = total_count - success_count
    if success_count == total_count:
        print("向量数据库入库全部完成！")
        logger.warning("向量数据库入库全部完成！")
    else:
        if success_count == 0:
            print("所有文件向量数据库入库均失败或跳过。")
            logger.warning("所有文件向量数据库入库均失败或跳过。")
        else:
            print(f"有 {remain} 个文件未完成向量数据库入库。")
            logger.warning(f"有 {remain} 个文件未完成向量数据库入库。")

    if failed_details:
        for fname, reason in failed_details:
            logger.warning(f"文件 {fname} 入库失败, 原因: {reason}")
        failed_files_only = [f for (f, _) in failed_details]
        logger.warning(f"下列文件向量数据库入库处理失败: {failed_files_only}")

if __name__ == "__main__":
    # 连接 Milvus
    vector_db_config = get_vector_db_config()
    connections.connect(
        alias="default",
        host=vector_db_config["host"],
        port=vector_db_config["port"],
        user=vector_db_config["user"],
        password=vector_db_config["password"]
    )
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()