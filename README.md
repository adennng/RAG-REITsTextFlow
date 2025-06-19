# RAG-REITsTextFlow

一个用于REITs公告PDF文档处理项目。

## 项目简介

本项目是一个专门用于处理基础设施公募REITs（Real Estate Investment Trusts）公告PDF文件的完整RAG数据处理管道。系统能够自动化地将PDF公告文档转换为结构化数据，能够检测表格、实现跨页表格拼接，并将表格内容还原为便于检索的文本信息。并构建向量数据库和 Elasticsearch 以支持智能检索与问答系统。

## 🚀 主要功能

### 核心处理流程
1. **PDF文件处理** - 批量识别和解析PDF文件
2. **文本提取** - 多进程提取PDF文本内容，区分矢量页和扫描页
3. **表格检测** - 智能检测跨页表格，支持矢量页和扫描页
4. **图像描述** - 使用AI模型自动描述表格和非表格图像
5. **内容合并** - 将表格描述合并到文本中
6. **文本分割** - 智能分割文本为适合检索的片段
7. **向量化** - 生成文本嵌入向量
8. **数据存储** - 同时支持Elasticsearch和Milvus向量数据库

### 技术特性
- 🔄 **多进程/多线程处理** - 高效处理大批量文档
- 🤖 **多AI模型支持** - 集成智谱AI和阿里云AI多模态模型
- 📊 **智能表格处理** - 自动检测、拼接和描述复杂表格
- 🖼️ **OCR图像识别** - 结合PaddleOCR和Tesseract
- 💾 **多数据库支持** - MySQL、Milvus、Elasticsearch
- 🔧 **模块化设计** - 每个步骤独立运行，便于调试和维护

## 🚀 使用方法

### 运行完整流程
```bash
python run_all_announcement_scripts.py
```

### 分步骤运行
```bash
# 步骤1：处理PDF文件
python scripts/step1_process_pdfs.py

# 步骤2：提取文本
python scripts/step2_extract_text_onlyvactor_multi_process.py

# 其他步骤按需运行...
```

## 📁 项目结构

```
RAG-REITsTextFlow/
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python依赖包
├── .env.example                        # 环境变量模板
├── .gitignore                          # Git忽略文件
├── run_all_announcement_scripts.py    # 主控制脚本
├── common_utils.py                     # 通用工具函数
├── db_config.py                        # 数据库配置
├── file_paths_config.py               # 文件路径配置
├── model_config.py                     # AI模型配置
├── 脚本说明.txt                        # 脚本功能说明
└── scripts/                           # 处理脚本文件夹
    ├── step1_process_pdfs.py           # PDF文件处理
    ├── step2_extract_text_onlyvactor_multi_process.py  # 文本提取
    ├── step3_1_detection_vactor_multi_process.py       # 矢量页表格检测
    ├── step3_2_table_detection_scan_multifile.py       # 扫描页表格检测
    ├── step4_*.py                      # 表格描述相关脚本
    ├── step5_merge_table_into_text.py  # 表格文本合并
    ├── step6_text_segmentation.py     # 文本分割
    ├── step7_text_embedding.py        # 文本向量化
    ├── step8_*.py                      # 数据库存储脚本
    └── 其他工具脚本...
```


