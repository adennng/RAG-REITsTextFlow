# db_config.py
import os
from typing import Dict, Any

def get_db_config() -> Dict[str, Any]:
    """
    返回 MySQL 数据库连接的配置信息。
    从环境变量读取敏感信息，如果环境变量不存在则使用默认值。
    """
    db_config = {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', 'your_password_here'),
        'database': os.getenv('DB_NAME', 'reits'),
        'charset': 'utf8mb4',
        'init_command': "SET SESSION collation_connection = 'utf8mb4_unicode_ci'"
    }
    return db_config

def get_db_announcement_config() -> Dict[str, Any]:
    """
    返回 MySQL 数据库announcement连接的配置信息。
    从环境变量读取敏感信息，如果环境变量不存在则使用默认值。
    """
    db_announcement_config = {
        'host': os.getenv('DB_ANNOUNCEMENT_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_ANNOUNCEMENT_PORT', 3306)),
        'user': os.getenv('DB_ANNOUNCEMENT_USER', 'root'),
        'password': os.getenv('DB_ANNOUNCEMENT_PASSWORD', 'your_password_here'),
        'database': os.getenv('DB_ANNOUNCEMENT_NAME', 'announcement'),
        'charset': 'utf8mb4',
        'init_command': "SET SESSION collation_connection = 'utf8mb4_unicode_ci'"
    }
    return db_announcement_config

def get_vector_db_config() -> Dict[str, Any]:
    """
    返回向量数据库（Milvus）的连接配置信息。
    从环境变量读取敏感信息，如果环境变量不存在则使用默认值。
    """
    vector_db_config = {
        'host': os.getenv('MILVUS_HOST', 'localhost'),
        'port': int(os.getenv('MILVUS_PORT', 19530)),
        'user': os.getenv('MILVUS_USER', 'root'),
        'password': os.getenv('MILVUS_PASSWORD', 'your_password_here')
    }
    return vector_db_config

def get_elasticsearch_config() -> Dict[str, Any]:
    """
    返回 Elasticsearch 数据库的连接配置信息。
    从环境变量读取敏感信息，如果环境变量不存在则使用默认值。
    """
    elasticsearch_config = {
        'host': os.getenv('ES_HOST', '127.0.0.1'),
        'port': int(os.getenv('ES_PORT', 9200)),
        'username': os.getenv('ES_USERNAME', 'elastic'),
        'password': os.getenv('ES_PASSWORD', 'your_password_here'),
        'scheme': os.getenv('ES_SCHEME', 'http')
    }
    return elasticsearch_config


