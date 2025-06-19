#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析公告文件并灌库的主控制脚本
"""

import os
import sys
import subprocess
from datetime import datetime

# 脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 主脚本日志文件，放在当前目录
MAIN_LOG = os.path.join(CURRENT_DIR, "run_all_announcement_scripts.log")


def timestamp():
    """返回当前时间字符串，格式 YYYY-MM-DD HH:MM:SS"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    """追加一条日志到主日志文件"""
    with open(MAIN_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp()}] {msg}\n")


def run_script(script_name: str, child_log_name: str):
    """
    执行单个子脚本：
      1. 记录开始执行
      2. 运行脚本并打印其终端输出
      3. 记录结束与返回码
      4. 读取子脚本日志中新内容，写入主日志
    """
    log(f"开始执行脚本：{script_name}")
    print(f"\n=== 开始执行 {script_name} ===")

    # 子脚本日志文件路径
    child_log_path = os.path.join(CURRENT_DIR, child_log_name)
    # 记录原日志大小，用于只读取新增内容
    offset = os.path.getsize(child_log_path) if os.path.exists(child_log_path) else 0

    # 调用子脚本（透传输出）
    proc = subprocess.run(
        [sys.executable, os.path.join(CURRENT_DIR, script_name)],
        cwd=CURRENT_DIR
    )
    ret = proc.returncode

    log(f"脚本 {script_name} 执行结束，返回码：{ret}")
    print(f"=== 脚本 {script_name} 执行结束，返回码：{ret} ===")

    # 读取并记录子脚本日志中新写入的内容
    if os.path.exists(child_log_path):
        with open(child_log_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            new_content = f.read().strip()
        if new_content:
            log(f"—— {script_name} 的工作日志新增内容 ——\n{new_content}\n—— 结束 ——")
        else:
            log(f"{script_name} 的日志中无新增内容。")
    else:
        log(f"{script_name} 的日志文件 {child_log_name} 未找到。")


# 按顺序定义所有子脚本及其日志文件名
SCRIPTS = [
    ("scripts/step1_process_pdfs.py", "process_pdfs.log"),
    ("scripts/step2_extract_text_onlyvactor_multi_process.py", "extract_text_onlyvactor.log"),
    ("scripts/step3_1_detection_vactor_multi_process.py", "table_processing_vactor_multifile.log"),
    ("scripts/step3_2_table_detection_scan_multifile.py", "table_processing_scan_multifile.log"),
    ("scripts/step4_1_1_describe_table_images_multi_thread.py", "table_detection.log"),
    ("scripts/step4_1_2_describe_table_images_multi_thread_second.py", "table_detection.log"),
    ("scripts/step4_2_1_describe_not_table_images_llm.py", "not_table_detection.log"),
    ("scripts/step4_2_2_describe_not_table_images_llm_second.py", "not_table_detection.log"),
    ("scripts/step5_merge_table_into_text.py", "merge.log"),
    ("scripts/step6_text_segmentation.py", "text_segmentation.log"),
    ("scripts/step7_text_embedding.py", "embedding.log"),
    ("scripts/step8_1_ingest_elasticsearch_data.py", "es_database.log"),
    ("scripts/step8_2_ingest_vector_database.py", "vector_database.log")    
]


def main():
    str = "-"*50 + "开始" + "-"*50
    log(f"{str}")
    log("主控制脚本运行开始")
    print(f"主脚本日志：{MAIN_LOG}")

    for script, logname in SCRIPTS:
        run_script(script, logname)

    log("全部子脚本执行完毕")
    print("\n所有子脚本已执行完毕。")
    print(f"主日志保存在：{MAIN_LOG}")


if __name__ == "__main__":
    main()
