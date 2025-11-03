#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_vector_database.py
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    IndexType
)
from db_config import get_vector_db_config

# 获取向量数据库配置
vector_db_config = get_vector_db_config()
HOST = vector_db_config['host']
PORT = vector_db_config['port']

# 集合名称和向量维度
COLLECTION_NAME = "reits_announcement"
EMBEDDING_DIM = 2048  # 根据您的 embedding 模型调整

# 1) 定义 Collection 的 Schema
# 注意保证字段顺序、类型、max_length 与 step8_ingest_vector_database.py 插入时一致
fields = [
    # id 字段 (int64, 非主键)
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=False, auto_id=False),

    # global_id 作为主键, 且允许超长文件名 => 设置较大max_length
    FieldSchema(name="global_id", dtype=DataType.VARCHAR, max_length=1024, is_primary=True),

    # chunk_id
    FieldSchema(name="chunk_id", dtype=DataType.INT64),

    # file_path
    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=2048),

    # date => 存字符串即可
    FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=64),

    # fund_code
    FieldSchema(name="fund_code", dtype=DataType.VARCHAR, max_length=256),

    # short_name
    FieldSchema(name="short_name", dtype=DataType.VARCHAR, max_length=256),

    # announcement_title
    FieldSchema(name="announcement_title", dtype=DataType.VARCHAR, max_length=1024),

    # doc_type_1
    FieldSchema(name="doc_type_1", dtype=DataType.VARCHAR, max_length=256),

    # doc_type_2
    FieldSchema(name="doc_type_2", dtype=DataType.VARCHAR, max_length=256),

    # announcement_link
    FieldSchema(name="announcement_link", dtype=DataType.VARCHAR, max_length=1024),

    # source_file
    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=1024),

    # page_num
    FieldSchema(name="page_num", dtype=DataType.VARCHAR, max_length=1024),

    # picture_path
    FieldSchema(name="picture_path", dtype=DataType.VARCHAR, max_length=2048),

    # char_count
    FieldSchema(name="char_count", dtype=DataType.INT64),

    # prev_chunks (可能存JSON或逗号分隔的字符串)
    FieldSchema(name="prev_chunks", dtype=DataType.VARCHAR, max_length=2048),

    # next_chunks
    FieldSchema(name="next_chunks", dtype=DataType.VARCHAR, max_length=2048),

    # text (正文), 可能较长, max_length=65535
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),

    # embedding 向量字段
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]

schema = CollectionSchema(fields, description="Collection for REITs text and embeddings, storing all fields from DB table text_segmentation_embedding.")

# 2) 连接到 Milvus
print("正在连接到 Milvus 集群...")
connections.connect(alias="default", host=HOST, port=PORT)
print("连接成功！")

# 3) 检查是否存在同名 Collection，如存在则删除
existing_collections = utility.list_collections()
if COLLECTION_NAME in existing_collections:
    print(f"集合 '{COLLECTION_NAME}' 已存在，正在删除...")
    coll = Collection(name=COLLECTION_NAME)
    coll.drop()
    print("删除成功！")
else:
    print(f"集合 '{COLLECTION_NAME}' 不存在，无需删除。")

# 4) 创建新的 Collection
print(f"正在创建集合: {COLLECTION_NAME} ...")
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"集合 '{COLLECTION_NAME}' 创建成功！")

# 5) 为 embedding 字段创建索引
print(f"正在为集合 '{COLLECTION_NAME}' 的 embedding 字段创建索引...")
index_params = {
    "index_type": IndexType.IVF_PQ,
    "metric_type": "L2",
    "params": {"nlist": 16384, "m": 64}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("索引创建成功！")

# 6) 输出当前集合列表
print("当前集群中的集合列表：")
print(utility.list_collections())
