# 调用大模型进行图片描述——openai SDK（新增裁剪功能）
import base64
import os
import sys
import time
from PIL import Image
from io import BytesIO
from openai import OpenAI  
from model_config import MODEL_CONFIG  # 引入配置文件

# 配置参数（新增裁剪参数）
TOP_CROP_PIXELS = 300    # 上边距裁剪像素（可调整）
BOTTOM_CROP_PIXELS = 300 # 下边距裁剪像素（可调整）

# 默认的大模型厂商和模型名称
DEFAULT_VENDOR = "zhipu"
DEFAULT_MODEL_NAME = "GLM-4V-Flash"

def crop_image(image_path: str) -> bytes:
    """
    裁剪图片去除页眉页脚区域
    :param image_path: 原始图片路径
    :return: 裁剪后的图片字节数据
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 计算裁剪区域
            crop_top = max(0, TOP_CROP_PIXELS)
            crop_bottom = max(0, height - BOTTOM_CROP_PIXELS)
            
            # 校验裁剪参数
            if crop_bottom <= crop_top:
                raise ValueError(
                    f"无效裁剪参数：上边距({crop_top}px) ≥ 有效高度({crop_bottom}px) | "
                    f"原始尺寸：{width}x{height}px"
                )
            
            # 执行裁剪并保存到内存
            cropped_img = img.crop((0, crop_top, width, crop_bottom))
            img_byte_arr = BytesIO()
            cropped_img.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
            
    except Exception as e:
        raise RuntimeError(f"图片裁剪失败：{str(e)}") from e

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
    调用指定的大模型对给定图片生成表格描述，
    以流式输出的方式显示生成内容，并在请求超时时自动重试指定次数。
    图片数据以 Base64 编码后通过 "image_url" 传递给大模型。
    
    :param image_path: 图片路径
    :param vendor: 大模型厂商（默认使用 DEFAULT_VENDOR）
    :param model_name: 大模型名称（默认使用 DEFAULT_MODEL_NAME）
    :param retries: 重试次数
    :param delay: 每次重试的间隔秒数
    :return: 图片描述文本
    """
    # 获取指定厂商和模型名称对应的配置
    model_config = get_model_config(vendor, model_name)

    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])

    try:
        # 新增裁剪处理步骤
        cropped_image_data = crop_image(image_path)
        base64_image = base64.b64encode(cropped_image_data).decode("utf-8")
    except Exception as e:
        raise Exception(f"图片预处理失败: {e}")

    # 构造提示词（保持原样）
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
        "   - 如果图片中包含文字（非表格），请逐段一字不落地描述文字内容，保留原文的段落结构和顺序。\n"
        "   - 如果判断本页的首行信息是某一段落的开始（例如首个字前有空格或缩进），或者判断本页首行为标题行，则在本页首个字前加上换行符“\n”。\n"
        "4. **限制条件**：\n"
        "   - 仅基于图片中的正文内容生成描述，禁止添加推测性信息。\n"
        "   - 严格按照图片中的原始顺序生成描述内容。\n"
    )

    # 构造消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ]

    # 重试机制与流式输出
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                stream=True
            )
            full_content = ""
            # 遍历流式响应，每个 chunk 为中间结果
            for chunk in response:
                try:
                    part = chunk.choices[0].delta.content
                    if part:
                        sys.stdout.write(part)
                        sys.stdout.flush()
                        full_content += part
                except Exception:
                    continue
            if full_content:
                sys.stdout.flush()
                return full_content
            else:
                sys.stdout.write(f"\n第 {attempt+1} 次请求未返回有效内容。\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"\n第 {attempt+1} 次请求异常: {e}\n")
            sys.stdout.flush()
        time.sleep(delay)
    raise Exception("OpenAI API 多次请求未返回有效流式输出，请稍后重试。")

if __name__ == "__main__":
    image_path = r"***.png"
    description = generate_table_description(image_path)