from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from io import BytesIO
import time
import os


def download_image(image_url):
    """从URL下载图片"""
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))


def save_image(image, filename):
    """保存图片到本地"""
    image.save(filename)
    print(f"图片已保存至 {filename}")
    return filename


def test_image_understanding(model, tokenizer, image_path, prompt):
    """测试模型的图像理解能力"""
    # 加载图片
    image = Image.open(image_path) if isinstance(image_path, str) else image_path

    # 构建输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 使用tokenizer处理多模态输入
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
        model.device
    )

    # 开始计时
    start_time = time.time()

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=512, temperature=0.7, do_sample=True
        )

    # 计算耗时
    elapsed_time = time.time() - start_time

    # 解码输出
    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)

    return response, elapsed_time


def test_multimodal_tasks(model, tokenizer):
    """测试多模态任务"""
    results = {}

    # 创建保存图片的文件夹
    if not os.path.exists("test_images"):
        os.makedirs("test_images")

    # 测试1: 图像描述
    print("\n===== 测试1: 图像描述 =====")
    image_url = "https://raw.githubusercontent.com/open-mmlab/mmdeploy/master/demo/resources/human-pose.jpg"
    image = download_image(image_url)
    image_path = save_image(image, "test_images/human_pose.jpg")

    prompt = "详细描述这张图片中的内容。"
    response, elapsed_time = test_image_understanding(
        model, tokenizer, image_path, prompt
    )

    print(f"提示: {prompt}")
    print(f"回答: {response}")
    print(f"生成耗时: {elapsed_time:.2f}秒")
    results["图像描述"] = {"response": response, "time": elapsed_time}

    # 测试2: 图像分析
    print("\n===== 测试2: 图像分析 =====")
    image_url = "https://raw.githubusercontent.com/QwenLM/Qwen-VL/main/assets/demo.jpg"
    image = download_image(image_url)
    image_path = save_image(image, "test_images/demo.jpg")

    prompt = "分析这张图片，并告诉我图中有哪些物体？人物在做什么？"
    response, elapsed_time = test_image_understanding(
        model, tokenizer, image_path, prompt
    )

    print(f"提示: {prompt}")
    print(f"回答: {response}")
    print(f"生成耗时: {elapsed_time:.2f}秒")
    results["图像分析"] = {"response": response, "time": elapsed_time}

    # 测试3: 图像中的文本识别
    print("\n===== 测试3: 图像中的文本识别 =====")
    image_url = "https://github.com/QwenLM/Qwen-VL/raw/main/assets/ocr.jpeg"
    image = download_image(image_url)
    image_path = save_image(image, "test_images/ocr.jpeg")

    prompt = "这张图片上写了什么文字？请提取并翻译成中文。"
    response, elapsed_time = test_image_understanding(
        model, tokenizer, image_path, prompt
    )

    print(f"提示: {prompt}")
    print(f"回答: {response}")
    print(f"生成耗时: {elapsed_time:.2f}秒")
    results["文本识别"] = {"response": response, "time": elapsed_time}

    # 测试4: 图像推理问题
    print("\n===== 测试4: 图像推理问题 =====")
    image_url = (
        "https://raw.githubusercontent.com/QwenLM/Qwen-VL/main/assets/reasoning.jpg"
    )
    image = download_image(image_url)
    image_path = save_image(image, "test_images/reasoning.jpg")

    prompt = "根据图片回答：这两个人可能是什么关系？他们在做什么？"
    response, elapsed_time = test_image_understanding(
        model, tokenizer, image_path, prompt
    )

    print(f"提示: {prompt}")
    print(f"回答: {response}")
    print(f"生成耗时: {elapsed_time:.2f}秒")
    results["图像推理"] = {"response": response, "time": elapsed_time}

    # 测试5: 视觉编程
    print("\n===== 测试5: 视觉编程 =====")
    image_url = (
        "https://raw.githubusercontent.com/QwenLM/Qwen-VL/main/assets/invoice.png"
    )
    image = download_image(image_url)
    image_path = save_image(image, "test_images/invoice.png")

    prompt = "请编写一个Python函数，可以从这种类型的发票图片中提取关键信息，如日期、金额和商品名称。"
    response, elapsed_time = test_image_understanding(
        model, tokenizer, image_path, prompt
    )

    print(f"提示: {prompt}")
    print(f"回答: {response}")
    print(f"生成耗时: {elapsed_time:.2f}秒")
    results["视觉编程"] = {"response": response, "time": elapsed_time}

    return results


def main():
    print("开始测试Qwen2.5-Omni INT8量化模型的多模态能力...")

    # 下载模型
    print("下载模型中...")
    model_dir = snapshot_download("qwen/Qwen2.5-Omni", revision="v1.0.0")
    print(f"模型已下载到: {model_dir}")

    # 配置INT8量化参数
    print("配置INT8量化参数...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16"
    )

    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # 加载量化后的模型
    print("加载INT8量化模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 打印显存使用情况
    if torch.cuda.is_available():
        used_memory = torch.cuda.max_memory_allocated() / (1024**3)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU显存使用: {used_memory:.2f}GB / {total_memory:.2f}GB")

    # 运行多模态测试
    print("\n开始多模态能力测试...")
    results = test_multimodal_tasks(model, tokenizer)

    # 打印总结报告
    print("\n===== 测试总结报告 =====")
    for task, data in results.items():
        print(f"{task}: 耗时 {data['time']:.2f}秒")

    # 最终显存使用情况
    if torch.cuda.is_available():
        used_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"测试完成后的最大显存使用: {used_memory:.2f}GB / {total_memory:.2f}GB")


if __name__ == "__main__":
    main()
