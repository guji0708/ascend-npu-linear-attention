#!/usr/bin/env python3
"""
农业智能诊断Demo - 作物病虫害识别与防治建议
基于Qwen3.5-0.8B在昇腾NPU上的多模态推理服务
界面风格：参考华为/Apple简约设计
"""
import gradio as gr
import time
import json
from pathlib import Path


def load_model(model_path):
    """加载Qwen3.5-0.8B模型（NPU环境）"""
    try:
        import torch
        import torch_npu
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = "npu:0"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor, device
    except ImportError:
        print("[WARN] NPU环境不可用，Demo将以模拟模式运行")
        return None, None, "cpu"


# 农业诊断模拟回复（本地无NPU时展示用）
SIMULATION_RESPONSES = {
    "default": (
        "[模拟诊断] 基于Qwen3.5-0.8B在昇腾NPU上的多模态分析：\n\n"
        "识别作物：水稻（粮食作物）\n"
        "诊断结果：观察到中度异常症状，初步判断可能为稻瘟病。\n\n"
        "诊断依据：叶片出现梭形病斑，中央灰白色，边缘褐色，符合稻瘟病典型症状。\n\n"
        "防治建议：推荐使用75%三环唑可湿性粉剂1000倍液预防，"
        "发病后用40%稻瘟灵乳油800倍液治疗。建议加强田间水位管理，"
        "适当增施硅肥提高抗病性，及时清除病株残体。\n\n"
        "风险提示：稻瘟病在高温高湿条件下传播迅速，建议立即用药并持续监测。"
    ),
    "healthy": (
        "[模拟诊断] 基于Qwen3.5-0.8B在昇腾NPU上的多模态分析：\n\n"
        "识别作物：苹果树（果树）\n"
        "诊断结果：叶片颜色正常，暂未发现明显病斑或虫害迹象。\n\n"
        "建议：继续保持良好的田间管理，定期巡查，"
        "注意预防苹果褐斑病、苹果黑星病等常见病害。"
        "建议每隔15天进行一次预防性喷药。"
    ),
}


def generate_response(model, processor, device, image, question, history):
    """生成诊断回复"""
    if model is None:
        time.sleep(0.8)
        keywords = question.lower() if question else ""
        if any(k in keywords for k in ["健康", "正常", "无病", "没病"]):
            return SIMULATION_RESPONSES["healthy"]
        return SIMULATION_RESPONSES["default"]

    import torch

    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": question})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    response = processor.decode(output[0], skip_special_tokens=True)
    return response


# ==================== 界面 ====================

CUSTOM_CSS = """
:root {
    --primary: #1B4332;
    --accent: #2D6A4F;
    --bg: #F0FDF4;
    --card: #FFFFFF;
    --text: #1A1A1A;
    --border: #D1FAE5;
}
.gradio-container {
    max-width: 920px !important;
    background: var(--bg);
    font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif;
}
.main-header {
    text-align: center;
    padding: 36px 0 16px;
}
.main-header h1 {
    font-size: 28px;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 14px;
    color: #6B7280;
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 500;
    background: #ECFDF5;
    color: #059669;
    margin-top: 8px;
}
.diagnosis-output {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-top: 8px;
}
"""

EXAMPLES = [
    [None, "请识别这张作物图片并判断是否有病虫害。"],
    [None, "作物叶片出现了黄斑，可能是什么病？"],
    [None, "这棵果树的生长状况如何？需要防治什么？"],
    [None, "请给出这个作物的健康管理建议。"],
]


def create_demo(model_path="./ckpt/hf_path/Qwen3.5-0.8B"):
    model, processor, device = load_model(model_path)

    def chat_fn(image, question, history):
        if not question:
            return history, "请输入诊断问题"
        response = generate_response(model, processor, device, image, question, history)
        history.append((question, response))
        return history, ""

    def clear_fn():
        return [], ""

    def quick_diagnose(image, history):
        if image is None:
            return history, "请先上传作物图片"
        question = "请识别这张图片中的作物种类，并判断是否存在病虫害迹象，给出防治建议。"
        response = generate_response(model, processor, device, image, question, history)
        history.append((question, response))
        return history, ""

    with gr.Blocks(css=CUSTOM_CSS, title="农业智能诊断 | 昇腾NPU") as demo:
        with gr.Column():
            gr.HTML("""
            <div class="main-header">
                <h1>农业智能诊断</h1>
                <p>作物病虫害识别与防治建议 · Powered by Qwen3.5-0.8B on Ascend NPU</p>
                <span class="status-badge">昇腾NPU加速</span>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="上传作物照片",
                        type="filepath",
                        height=320,
                    )
                    diagnose_btn = gr.Button(
                        "一键诊断", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="诊断对话",
                        height=400,
                        bubble_full_width=False,
                    )
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="输入问题",
                            placeholder="描述作物症状或提出问题...",
                            scale=4,
                            lines=1,
                        )
                        send_btn = gr.Button("发送", scale=1, variant="primary")

                    with gr.Row():
                        clear_btn = gr.Button("清空对话", scale=1)
                        examples = gr.Examples(
                            examples=EXAMPLES,
                            inputs=[image_input, question_input],
                        )

        diagnose_btn.click(
            quick_diagnose,
            inputs=[image_input, chatbot],
            outputs=[chatbot, question_input],
        )
        send_btn.click(
            chat_fn,
            inputs=[image_input, question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        question_input.submit(
            chat_fn,
            inputs=[image_input, question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        clear_btn.click(clear_fn, outputs=[chatbot, question_input])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="农业智能诊断Demo")
    parser.add_argument("--model", default="./ckpt/hf_path/Qwen3.5-0.8B", help="模型路径")
    parser.add_argument("--port", type=int, default=7860, help="端口")
    args = parser.parse_args()

    demo = create_demo(args.model)
    demo.launch(server_name="0.0.0.0", server_port=args.port)
