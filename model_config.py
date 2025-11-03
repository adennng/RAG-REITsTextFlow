# model_config.py
MODEL_CONFIG = {
    "deepseek": {
         "deepseek-chat": {
                  "model": "deepseek-chat",
                  "api_key": "YOUR_DEEPSEEK_API_KEY",
                  "base_url": "https://api.deepseek.com"
         },
         "deepseek-reasoner": {
                  "model": "deepseek-reasoner",
                  "api_key": "YOUR_DEEPSEEK_API_KEY",
                  "base_url": "https://api.deepseek.com"
         }
    },
    "zhipu": {
         "GLM-4V-Flash": {
                    "model": "GLM-4V-Flash",
                    "api_key": "YOUR_ZHIPU_API_KEY",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4/"
         },
         "embedding-3": {
                    "model": "embedding-3",
                    "api_key": "YOUR_ZHIPU_API_KEY",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4/"
         }
    },
    "ali": {
         "qwen-vl-max-latest": {
                  "model": "qwen-vl-max-latest",
                  "api_key": "YOUR_ALI_API_KEY",
                  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
         },
         "qwen-vl-max-2025-01-25": {
                  "model": "qwen-vl-max-2025-01-25",
                  "api_key": "YOUR_ALI_API_KEY",
                  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
         },
         "deepseek-v3": {
                  "model": "deepseek-v3",
                  "api_key": "YOUR_ALI_API_KEY",
                  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
         },
         "deepseek-r1": {
                  "model": "deepseek-r1",
                  "api_key": "YOUR_ALI_API_KEY",
                  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
         }
    }
}


