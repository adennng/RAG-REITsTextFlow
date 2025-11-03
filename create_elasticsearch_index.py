#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_elasticsearch_index.py
"""

from elasticsearch import Elasticsearch
from db_config import get_elasticsearch_config

es_config = get_elasticsearch_config()

es = Elasticsearch(
    [f"{es_config.get('scheme', 'http')}://{es_config['host']}:{es_config['port']}"],
    basic_auth=(es_config['username'], es_config['password']),
    verify_certs=False
)

index_name = "reits_announcements"

# 为与数据库 text_segmentation_embedding 对应，这里列出相应字段
index_body = {
    "mappings": {
        "properties": {
            "id": {"type": "long"},
            "global_id": {"type": "keyword"},
            "chunk_id": {"type": "long"},
            "file_path": {"type": "keyword"},
            "date": {"type": "date"},   # 需保证插入时是 'yyyy-MM-dd' 或类似
            "fund_code": {"type": "keyword"},
            "short_name": {"type": "text"},
            "announcement_title": {"type": "text"},
            "doc_type_1": {"type": "keyword"},
            "doc_type_2": {"type": "keyword"},
            "announcement_link": {"type": "keyword"},
            "source_file": {"type": "keyword"},
            "page_num": {"type": "keyword"},
            "picture_path": {"type": "keyword"},
            "char_count": {"type": "integer"},
            "prev_chunks": {"type": "keyword"},
            "next_chunks": {"type": "keyword"},
            "text": {"type": "text"} 
            # 不存 embedding 到 ES, 由 Milvus 做向量检索
        }
    }
}

# 1. 删除已存在的同名索引
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"已删除已存在的索引: {index_name}")

# 2. 创建新索引
es.indices.create(index=index_name, body=index_body)
print(f"索引 '{index_name}' 创建成功！")
