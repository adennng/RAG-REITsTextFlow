# step4_table_utils_ali.py
# 调用大模型进行图片描述——DashScope SDK
import base64
import os
import sys
import time
import dashscope
from model_config import MODEL_CONFIG  # 引入配置文件

# 默认的大模型厂商和模型名称
DEFAULT_VENDOR = "ali"
DEFAULT_MODEL_NAME = "qwen-vl-max-2025-01-25"


def get_model_config(vendor: str, model_name: str) -> dict:
    """
    获取指定厂商和模型名称的配置。
    :param vendor: 大模型厂商
    :param model_name: 大模型名称
    :return: 对应的模型配置信息字典
    """
    if vendor not in MODEL_CONFIG:
        raise ValueError(f"厂商 {vendor} 不存在于配置文件中。")
    
    vendor_config = MODEL_CONFIG[vendor]
    
    if model_name not in vendor_config:
        raise ValueError(f"模型 {model_name} 不存在于厂商 {vendor} 的配置中。")
    
    return vendor_config[model_name]


def generate_table_description(
    image_path: str,
    vendor: str = DEFAULT_VENDOR,
    model_name: str = DEFAULT_MODEL_NAME,
    retries: int = 3,
    delay: int = 5
) -> str:
    """
    调用 DashScope SDK 以流式输出的方式对给定图片生成表格描述，
    并采用连续输出的方式构造返回值。如果请求超时，将自动重试指定次数。
    
    :param image_path: 图片路径
    :param vendor: 大模型厂商（默认使用 DEFAULT_VENDOR）
    :param model_name: 大模型名称（默认使用 DEFAULT_MODEL_NAME）
    :param retries: 重试次数
    :param delay: 每次重试间隔秒数
    :return: 表格描述文本
    """
    # 获取模型配置
    model_config = get_model_config(vendor, model_name)

    # 读取图片并转换为 Base64 编码
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    except Exception as e:
        raise Exception(f"读取图片失败: {e}")

    base64_image = base64.b64encode(image_data).decode("utf-8")

    # 构造文本提示
    text_prompt = (
        "请根据以下规则描述图片中的内容：\n"
        "1. **描述范围**：\n"
        "   - 只描述图片中的正文内容，忽略页眉、页脚、页码等非正文信息。\n"
        "   - 如果包含文字，请完整描述文字内容；如果图片中包含表格，请完整描述表格内容。\n"
        "2. **表格描述规则**：\n"
        "   - 如果图片中包含表格，请按照以下格式描述：\n"
        "     ## 表格主题\n"
        "     - 明确表格的主题（例如“XX项目历史运营数据”或“XX指标历史趋势”）。\n"
        "     - 如果表格有编号，请包含编号。\n"
        "     - 示例：“表格展示了基础设施项目2021至2024年的历史租金收缴率。”\n"
        "     ## 表格内容\n"
        "     - 按“横向行”逐行描述，对每个指标，按列顺序（从左到右）说明各时间段的具体数值，需包含单位和百分比。"
        "     - 示例：“表格内容为：出租率在2024年1-6月为100.00%，2023年度为100.00%，2022年度为100.00%，2021年度为100.00%”。\n"
        "     ## 描述结尾\n"
        "     - 在表格内容后面加上“表格内容描述完毕。”\n"
        "     ##格式要求：\n"
        "     - 请注意！使用纯文本段落，禁止使用表格、列表符号，如“|”、“---”等。\n"
        "     - 数值需与指标名称直接关联，避免歧义（例如“2023年度运营收入（不含税）为2,194.72万元”）。\n"
        "3. **文字描述规则**：\n"
        "   - 如果图片中包含文字（非表格），请逐段一字不落的描述文字内容，保留原文的段落结构和顺序。\n"
        "   - 如果判断本页的首行信息是某一段落的开始（例如首个字前面有空格或缩进）、或者你判断本页的首行为标题行，则请在本页首个字的前面加上换行符“\n”。\n"
        "4. **限制条件**：\n"
        "   - 请注意！不要遗漏正文中任何信息！禁止合并相关内容！生成完毕检查是否有信息没有被描述到，如果有，请补充。\n"
        "   - 仅基于图片中的正文内容生成，禁止添加推测性信息。\n"
        "   - 严格按照图片中的原始顺序生成描述内容。\n"
    )

    # 构造消息（图片信息和文本提示）
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{base64_image}"},
                {"text": text_prompt}
            ]
        }
    ]

    # 重试机制
    for attempt in range(retries):
        responses = dashscope.MultiModalConversation.call(
            api_key=model_config["api_key"],
            model=model_config["model"],
            messages=messages,
            stream=True,
            incremental_output=True,
            vl_high_resolution_images=True
        )
        full_content = ""
        try:
            # 遍历流式响应，每个 response 为一个中间结果
            for response in responses:
                try:
                    chunk = response["output"]["choices"][0]["message"].content[0]["text"]
                    # =============================
                    # 注释掉原来输出到终端的代码：
                    # sys.stdout.write(chunk)
                    # sys.stdout.flush()
                    # 但仍保留拼接到 full_content
                    # =============================
                    full_content += chunk
                except Exception:
                    continue

            if full_content:
                # 注释掉原来刷新输出的代码
                # sys.stdout.flush()
                return full_content
            else:
                print(f"\n第 {attempt+1} 次请求未返回有效内容。")
        except Exception as e:
            print(f"\n第 {attempt+1} 次请求异常: {e}")
        time.sleep(delay)

    raise Exception("DashScope API 多次请求未返回有效流式输出，请稍后重试。")


if __name__ == "__main__":
    image_path = r"***.png"
    # 终端不再输出大模型返回内容，但函数依旧返回描述文本
    desc = generate_table_description(image_path)
    print("描述生成完毕，不在终端显示大模型内容。\n")
