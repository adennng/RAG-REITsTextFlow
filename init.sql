-- 初始化数据库脚本

-- 创建 reits 数据库
CREATE DATABASE IF NOT EXISTS reits CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建 announcement 数据库
CREATE DATABASE IF NOT EXISTS announcement CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用 announcement 数据库
USE announcement;

-- 创建 processed_files 表
CREATE TABLE IF NOT EXISTS processed_files (
    file_name VARCHAR(255) PRIMARY KEY COMMENT '文件名（主键）',
    file_path TEXT COMMENT '文件路径',
    date DATE COMMENT '公告日期',
    fund_code VARCHAR(20) COMMENT '基金代码',
    short_name VARCHAR(100) COMMENT '基金简称',
    announcement_title TEXT COMMENT '公告标题',
    doc_type_1 VARCHAR(50) COMMENT '公告类型一级',
    doc_type_2 VARCHAR(50) COMMENT '公告类型二级',
    announcement_link TEXT COMMENT '公告链接',
    text_extracted VARCHAR(10) DEFAULT 'false' COMMENT '文本提取完成',
    table_detection_vector_done VARCHAR(10) DEFAULT 'false' COMMENT '矢量页表格检测完成',
    table_detection_scan_done VARCHAR(10) DEFAULT 'false' COMMENT '扫描页表格检测完成',
    table_describe_done VARCHAR(10) DEFAULT 'false' COMMENT '表格描述完成',
    not_table_describe_done VARCHAR(10) DEFAULT 'false' COMMENT '非表格图像描述完成',
    merge_done VARCHAR(10) DEFAULT 'false' COMMENT '内容合并完成',
    text_segmentation VARCHAR(10) DEFAULT 'false' COMMENT '文本分割完成',
    embedding_done VARCHAR(10) DEFAULT 'false' COMMENT '向量化完成',
    vector_database_done VARCHAR(10) DEFAULT 'false' COMMENT '向量数据库存储完成',
    elasticsearch_database_done VARCHAR(10) DEFAULT 'false' COMMENT 'Elasticsearch存储完成',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) ENGINE=InnoDB COMMENT='已处理文件表';

-- 创建文本分割表
CREATE TABLE IF NOT EXISTS text_segmentation_embedding (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    file_name VARCHAR(255) NOT NULL COMMENT '文件名',
    segment_id INT NOT NULL COMMENT '分段ID',
    text TEXT NOT NULL COMMENT '分段文本',
    embedding TEXT COMMENT '向量嵌入（JSON格式）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_file_name (file_name),
    INDEX idx_segment_id (segment_id)
) ENGINE=InnoDB COMMENT='文本分割和向量嵌入表';

-- 使用 reits 数据库创建示例表（根据实际需求调整）
USE reits;

-- 创建公告信息表（示例表结构，需要根据实际情况调整）
CREATE TABLE IF NOT EXISTS 公告信息 (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    公告标题 VARCHAR(500) NOT NULL COMMENT '公告标题',
    基金代码 VARCHAR(20) NOT NULL COMMENT '基金代码',
    公告日期 DATE NOT NULL COMMENT '公告日期',
    公告类型_一级 VARCHAR(50) COMMENT '公告类型一级分类',
    公告类型_二级 VARCHAR(50) COMMENT '公告类型二级分类',
    公告链接 TEXT COMMENT '公告原始链接',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_title_code_date (公告标题(100), 基金代码, 公告日期)
) ENGINE=InnoDB COMMENT='公告基本信息表';