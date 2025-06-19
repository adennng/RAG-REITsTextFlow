# file_paths_config.py
# 文件路径配置 - 支持环境变量配置
import os

# PDF 文件目录路径
PDF_DIR = os.getenv('PDF_DIR', './data/pdfs/')

# 输出文件夹路径
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './data/output/')

# Table Transformer模型路径
table_transformer_path = os.getenv('TABLE_TRANSFORMER_PATH', './models/table-transformer-detection')

