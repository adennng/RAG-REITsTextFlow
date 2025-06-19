# model_config.py
#多模态模型GLM-4V-Flash模型免费
#多模态模型中，qwen-vl-max-2025-01-25图片识别效果及遵从提示词效果最佳

import os
from typing import Dict, Any



def get_model_config() -> Dict[str, Any]:
    """
    返回模型配置信息。
    从环境变量读取API密钥等敏感信息。
    """
    MODEL_CONFIG = {
        "zhipu": {
             "GLM-4V-Flash": {
                        "model": "GLM-4V-Flash",
                        "api_key": os.getenv("ZHIPU_API_KEY", "your_zhipu_api_key_here"),
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/"
             },
             "embedding-3": {
                        "model": "embedding-3",
                        "api_key": os.getenv("ZHIPU_API_KEY", "your_zhipu_api_key_here"),
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/"
             }
        },
        "ali": {
             "qwen-vl-max-latest": {
                      "model": "qwen-vl-max-latest",
                      "api_key": os.getenv("ALI_API_KEY", "your_ali_api_key_here"),
                      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
             },
             "qwen-vl-max-2025-01-25": {
                      "model": "qwen-vl-max-2025-01-25",
                      "api_key": os.getenv("ALI_API_KEY", "your_ali_api_key_here"),
                      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
             },
             "deepseek-v3": {
                      "model": "deepseek-v3",
                      "api_key": os.getenv("ALI_API_KEY", "your_ali_api_key_here"),
                      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
             },
             "deepseek-r1": {
                      "model": "deepseek-r1",
                      "api_key": os.getenv("ALI_API_KEY", "your_ali_api_key_here"),
                      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
             }
        }
    }
    return MODEL_CONFIG

# 为了向后兼容，保留全局变量
MODEL_CONFIG = get_model_config()


