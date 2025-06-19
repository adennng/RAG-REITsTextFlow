import os
from pymilvus import connections

def get_vector_db_config():
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

def main():
    config = get_vector_db_config()
    try:
        connections.connect(alias="default", host=config['host'], port=config['port'])
        print("成功连接到 Milvus!")
    except Exception as e:
        print("连接 Milvus 失败:", e)

if __name__ == "__main__":
    main()
