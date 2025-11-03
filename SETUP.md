# 项目配置指南

本文档详细说明如何配置 RAG-REITsTextFlow 项目。

## 1. 数据库配置

### 1.1 编辑 `db_config.py`

打开 `db_config.py` 文件,替换以下占位符为您的真实配置:

```python
# MySQL 数据库配置
'password': 'YOUR_MYSQL_PASSWORD'  # 替换为您的MySQL密码

# Elasticsearch 配置
'password': 'YOUR_ELASTICSEARCH_PASSWORD'  # 替换为您的Elasticsearch密码

# Milvus 向量数据库配置
'password': 'YOUR_MILVUS_PASSWORD'  # 替换为您的Milvus密码

# 远程数据库配置(如需要)
'host': 'YOUR_REMOTE_HOST'  # 替换为远程数据库地址
'password': 'YOUR_REMOTE_DB_PASSWORD'  # 替换为远程数据库密码
```

## 2. AI模型API配置

### 2.1 编辑 `model_config.py`

打开 `model_config.py` 文件,根据您使用的AI服务商,替换对应的API密钥:

#### DeepSeek API
```python
"api_key": "YOUR_DEEPSEEK_API_KEY"
```
申请地址: https://platform.deepseek.com/

#### 智谱AI API
```python
"api_key": "YOUR_ZHIPU_API_KEY"
```
申请地址: https://open.bigmodel.cn/

#### 阿里云通义千问 API
```python
"api_key": "YOUR_ALI_API_KEY"
```
申请地址: https://dashscope.aliyun.com/


## 3. 文件路径配置

### 3.1 编辑 `file_paths_config.py`

```python
# PDF 文件输入目录
PDF_DIR = r"/your/path/to/pdfs/"

# 处理结果输出目录
OUTPUT_DIR = r"/your/path/to/output/"

# Table Transformer 模型路径
table_transformer_path = r"/your/path/to/table-transformer-detection"

```

## 4. 数据库初始化

### 4.1 MySQL 数据库


创建数据库 announcement 建立相关表



### 4.2 Elasticsearch

```bash
# 安装Elasticsearch后运行运行创建索引脚本
python create_elasticsearch_index.py
```

### 4.3 Milvus 向量数据库

```bash

# 安装Milvus后运行创建向量库脚本
python create_vector_database.py
```

