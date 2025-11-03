#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 将 mysql 数据库中表 text_segmentation_embedding 的 text 转为向量保存至 “embedding” 字段
"""
step7_text_embedding.py (多线程版本, 记录失败原因)

1) 使用 ThreadPoolExecutor 并发处理文件
2) 对每个文件:
   - 获取 db 连接 => 开启事务 => 查询 text_segmentation_embedding => 生成 embedding => 更新 => 提交
   - 若失败 => rollback => 记录错误原因
3) 在 future.result() 成功后, 更新数据库 processed_files 表的 embedding_done 状态
4) 对于失败文件, 在日志 embedding.log 中记录其文件名及失败原因
5) 对于空文本块, 默认不视为错误(只要 API 不报错即可). 如果全部 chunk 都为空, 仍会送空字符串到 API (不会报错时即不是错误).
   若想特别处理可在 embedding 前检测.
"""

import os
import re
import json
import time
import logging
import pymysql
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------ import local configs ------------------
import file_paths_config
from db_config import get_db_announcement_config
from model_config import MODEL_CONFIG
from openai import OpenAI  # 封装 zhipu API
from common_utils import safe_json_dump

MULTIFILE_OUTPUT_DIR = file_paths_config.OUTPUT_DIR

# 日志文件 embedding.log, WARNING 及以上级别
# 获取脚本所在目录，确保日志文件生成在log目录下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, "embedding.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
file_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 并发线程数，可按需修改
MAX_WORKERS = 5

# embedding 相关配置
EMBED_BATCH_SIZE = 64
MAX_CHARS = 3000
EMBEDDING_DIM = 2048

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
    从数据库获取需要进行向量化的文件
    条件: text_segmentation='true' AND embedding_done='false' AND doc_type_1 != '无关'
    返回: 按fund_code分组的文件列表字典
    """
    try:
        conn = get_announcement_connection()
        with conn.cursor() as cursor:
            sql = """
            SELECT file_name, file_path, date, fund_code, short_name, announcement_title,
                   doc_type_1, doc_type_2, announcement_link, text_segmentation, embedding_done
            FROM processed_files 
            WHERE text_segmentation='true' 
              AND embedding_done='false' 
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

class Config:
    embedding_provider = "zhipu"
    embedding_model = "embedding-3"

class OpenAIEmbeddings:
    """
    使用 Zhipu API 生成文本向量
    """
    def __init__(self):
        try:
            self.model_config = MODEL_CONFIG[Config.embedding_provider][Config.embedding_model]
        except KeyError:
            raise ValueError(f"未找到配置：{Config.embedding_provider}/{Config.embedding_model}")
        self.client = OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["base_url"]
        )
        self.model_name = self.model_config["model"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i:i+EMBED_BATCH_SIZE]
            # 截断
            batch_texts = [t[:MAX_CHARS] for t in batch_texts]
            print(f"正在生成向量 batch {i // EMBED_BATCH_SIZE + 1}, 文本数量: {len(batch_texts)}")
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    encoding_format="float"
                )
            except Exception as e:
                logger.warning(f"向量生成API调用失败: {e}")
                raise
            for item in response.data:
                all_embeddings.append(item.embedding)
        return all_embeddings

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, path):
    safe_json_dump(data, path)

def fetch_chunks_for_file(db_conn, source_file):
    """
    从数据库 text_segmentation_embedding 表中读取指定 source_file 的文本块
    返回 [ (id, global_id, chunk_id, text) ] 按 chunk_id 排序
    """
    sql = """
    SELECT id, global_id, chunk_id, text
    FROM text_segmentation_embedding
    WHERE source_file = %s
    ORDER BY chunk_id
    """
    with db_conn.cursor() as cursor:
        cursor.execute(sql, (source_file,))
        rows = cursor.fetchall()
    return rows

def clear_old_embeddings(db_conn, source_file):
    """
    清空该文件的 embedding 字段(设为 NULL)
    """
    sql = """
    UPDATE text_segmentation_embedding
    SET embedding = NULL
    WHERE source_file = %s
    """
    with db_conn.cursor() as cursor:
        cursor.execute(sql, (source_file,))

def update_embeddings_for_chunks(db_conn, chunks, embeddings):
    """
    按照 id 逐条更新 embedding 字段
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks数量与embedding结果数量不匹配")
    sql = """
    UPDATE text_segmentation_embedding
    SET embedding = %s
    WHERE id = %s
    """
    with db_conn.cursor() as cursor:
        import json
        for (row_id, _, _, _), vector in zip(chunks, embeddings):
            embedding_json = json.dumps(vector)
            cursor.execute(sql, (embedding_json, row_id))

def process_file_embedding(pdf_info):
    """
    针对单个文件的 embedding 处理逻辑(供线程池调用):
      1) 与数据库建立连接, 开启事务
      2) 根据 source_file 查询所有文本块
      3) 清空旧 embedding
      4) 生成 embedding
      5) 逐条更新 embedding
      6) 全部成功则 commit, 否则 rollback
      7) 返回 (success, error_reason) => 由主线程决定日志 & 标记
    """
    source_file_val = pdf_info.get("file_name", "")
    if not source_file_val:
        return (False, "文件信息中缺少 file_name")

    db_conf = get_db_announcement_config()
    connection = None
    try:
        connection = pymysql.connect(**db_conf)
        connection.begin()

        # 查询文本块
        chunks = fetch_chunks_for_file(connection, source_file_val)
        if not chunks:
            connection.rollback()
            return (False, f"数据库中未找到 source_file={source_file_val} 的文本块记录")

        # 清空旧 embedding
        clear_old_embeddings(connection, source_file_val)

        # 生成 embedding
        texts = [r[3] for r in chunks]  # r=(id, global_id, chunk_id, text)
        embedder = OpenAIEmbeddings()
        embeddings = embedder.embed_documents(texts)

        # 更新 embedding
        update_embeddings_for_chunks(connection, chunks, embeddings)

        connection.commit()
        return (True, None)
    except Exception as e:
        if connection:
            connection.rollback()
        return (False, str(e))
    finally:
        if connection:
            connection.close()

def main():
    processed_files = get_pending_files_from_db()
    if not processed_files:
        print("没有找到需要处理的文件。")
        logger.warning("没有找到需要处理的文件。")
        return

    # 遍历数据库查询结果，构建待处理列表
    files_to_process = []
    for fund_code, pdf_list in processed_files.items():
        for pdf_info in pdf_list:
            files_to_process.append(pdf_info)

    total_count = len(files_to_process)
    if total_count == 0:
        print("没有找到需要处理的文件。")
        logger.warning("没有找到需要处理的文件。")
        return

    print(f"本次需要处理 {total_count} 个文件的向量化...")
    success_count = 0
    failed_details = []  # 存 (file_name, reason)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {}
        for pdf_info in files_to_process:
            fut = executor.submit(process_file_embedding, pdf_info)
            future_to_file[fut] = pdf_info

        for fut in as_completed(future_to_file):
            pdf_info = future_to_file[fut]
            file_name = pdf_info.get("file_name", "")
            try:
                success, error_reason = fut.result()
            except Exception as e:
                success = False
                error_reason = str(e)
                logger.warning(f"文件 {file_name} embedding出现未知异常: {error_reason}")

            if success:
                # 更新数据库 processed_files 表，仅更新 embedding_done 字段
                try:
                    update_pdf_status_in_db(file_name, "embedding_done", "true")
                    success_count += 1
                    print(f"已完成 {file_name} 的向量化处理。")
                except Exception as e:
                    print(f"文件 {file_name} 更新数据库 processed_files 失败: {e}")
                    logger.warning(f"文件 {file_name} 更新数据库 processed_files 失败: {e}")
                    failed_details.append((file_name, f"更新数据库状态失败: {e}"))
            else:
                if not error_reason:
                    error_reason = "Unknown failure"
                print(f"{file_name} 处理向量化失败，原因: {error_reason}")
                failed_details.append((file_name, error_reason))

    remain = total_count - success_count
    if success_count == total_count:
        print("embedding全部完成！")
        logger.warning("embedding全部完成！")
    else:
        if success_count == 0:
            print("所有文件向量化均失败或跳过。")
            logger.warning("所有文件向量化均失败或跳过。")
        else:
            print(f"有 {remain} 个文件未完成向量化。")
            logger.warning(f"有 {remain} 个文件未完成向量化。")

    if failed_details:
        for (fname, reason) in failed_details:
            logger.warning(f"文件 {fname} 向量化处理失败，原因: {reason}")
        failed_files_only = [f for (f, _) in failed_details]
        logger.warning(f"下列文件向量化处理失败: {failed_files_only}")

if __name__ == "__main__":
    main()

# 强制刷新日志，放在脚本最后一行
import logging
logging.shutdown()