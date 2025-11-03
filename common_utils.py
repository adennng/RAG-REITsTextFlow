#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
common_utils.py - 通用工具模块

提供统一的JSON序列化功能，解决datetime对象序列化问题
"""

import json
import datetime
from typing import Any

class SafeJSONEncoder(json.JSONEncoder):
    """安全的JSON编码器，处理各种Python对象序列化"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, datetime.time):
            return obj.strftime('%H:%M:%S')
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象
            return obj.__dict__
        return super().default(obj)

def safe_json_dump(data: Any, file_path: str, **kwargs) -> None:
    """安全的JSON文件写入"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=SafeJSONEncoder, ensure_ascii=False, indent=2, **kwargs)

def safe_json_dumps(data: Any, **kwargs) -> str:
    """安全的JSON字符串序列化"""
    return json.dumps(data, cls=SafeJSONEncoder, ensure_ascii=False, **kwargs)

def safe_json_load(file_path: str) -> Any:
    """安全的JSON文件读取"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def safe_json_loads(json_str: str) -> Any:
    """安全的JSON字符串反序列化"""
    return json.loads(json_str) 