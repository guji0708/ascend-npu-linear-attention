#!/usr/bin/env python3
"""
COCO2017数据预处理脚本 - 农业病虫害诊断方向
将COCO2017标注转换为MindSpeed-MM的LLaVA-Instruct格式训练JSON

输入: COCO2017数据集目录（含images/和annotations/）
输出: mllm_format_llava_instruct_data.json
"""
import json
import os
import random
from pathlib import Path
from pycocotools.coco import COCO


# COCO类别中的农业相关动植物映射
CROP_CATEGORY_MAP = {
    # 水果类
    "apple": {"crop": "苹果树", "type": "果树", "common_diseases": ["苹果褐斑病", "苹果黑星病", "苹果轮纹病"]},
    "banana": {"crop": "香蕉树", "type": "热带果树", "common_diseases": ["香蕉叶斑病", "香蕉枯萎病", "香蕉黑星病"]},
    "orange": {"crop": "柑橘树", "type": "果树", "common_diseases": ["柑橘溃疡病", "柑橘黄龙病", "柑橘炭疽病"]},
    # 蔬菜类
    "broccoli": {"crop": "西兰花", "type": "十字花科蔬菜", "common_diseases": ["黑腐病", "霜霉病", "菌核病"]},
    "carrot": {"crop": "胡萝卜", "type": "根茎类蔬菜", "common_diseases": ["黑斑病", "软腐病", "根结线虫病"]},
    # 谷物类
    "rice": {"crop": "水稻", "type": "粮食作物", "common_diseases": ["稻瘟病", "纹枯病", "白叶枯病"]},
    "wheat": {"crop": "小麦", "type": "粮食作物", "common_diseases": ["锈病", "赤霉病", "白粉病"]},
    "corn": {"crop": "玉米", "type": "粮食作物", "common_diseases": ["大斑病", "小斑病", "锈病"]},
    # 经济作物
    "potted plant": {"crop": "观赏植物", "type": "园艺作物", "common_diseases": ["白粉病", "灰霉病", "叶斑病"]},
    "vase": {"crop": "观赏花卉", "type": "园艺作物", "common_diseases": ["灰霉病", "根腐病", "病毒病"]},
    # 畜牧相关
    "cow": {"crop": "肉牛/奶牛", "type": "畜牧", "common_diseases": ["口蹄疫", "乳房炎", "结核病"]},
    "sheep": {"crop": "绵羊/山羊", "type": "畜牧", "common_diseases": ["羊痘", "寄生虫病", "肺炎"]},
    "horse": {"crop": "马匹", "type": "畜牧", "common_diseases": ["马流感", "蹄叶炎", "寄生虫病"]},
}

# 通用作物映射（非特定类别的图片用）
GENERIC_CROP = {"crop": "农作物", "type": "通用作物", "common_diseases": ["叶斑病", "枯萎病", "锈病"]}

# 农业诊断场景的指令模板
INSTRUCTION_TEMPLATES = [
    {
        "user": "请识别这张图片中的作物种类，并判断是否存在病虫害迹象。",
        "assistant_prefix": "经多模态分析，图片中的作物为",
    },
    {
        "user": "这张图片中的作物叶片状态如何？有无异常？",
        "assistant_prefix": "基于图像特征分析，该",
    },
    {
        "user": "请对图片中的作物进行健康诊断。",
        "assistant_prefix": "作物健康诊断结果：识别到",
    },
    {
        "user": "图片中的作物可能患有什么病害？请给出防治建议。",
        "assistant_prefix": "根据图片症状特征，该",
    },
    {
        "user": "请分析图片中作物的生长状况和潜在风险。",
        "assistant_prefix": "生长状况分析：图中",
    },
]

# 防治建议库
TREATMENT_ADVICE = {
    "叶斑病": "建议喷施70%代森锰锌可湿性粉剂500倍液，每7-10天一次，连续2-3次。同时清除病叶，减少侵染源。",
    "锈病": "建议使用15%三唑酮可湿性粉剂1500倍液或25%丙环唑乳油2000倍液喷雾，发病初期用药效果最佳。",
    "白粉病": "推荐使用25%嘧菌酯悬浮剂1500倍液或50%醚菌酯水分散粒剂3000倍液，注意通风降湿。",
    "霜霉病": "建议喷施68.75%氟菌·霜霉威悬浮剂600倍液，保持田间通风，避免过度浇水。",
    "黑斑病": "使用10%苯醚甲环唑水分散粒剂1500倍液喷雾，配合清除病残体，加强田间管理。",
    "枯萎病": "建议用50%多菌灵可湿性粉剂500倍液灌根，实行轮作，选择抗病品种。",
    "灰霉病": "推荐使用50%腐霉利可湿性粉剂1000倍液或40%嘧霉胺悬浮剂800倍液，注意控温控湿。",
    "菌核病": "使用50%乙烯菌核利可湿性粉剂1000倍液喷雾，及时清除病株，深翻土壤。",
    "软腐病": "建议使用72%农用链霉素可溶性粉剂3000倍液喷洒，避免机械损伤，注意排水。",
    "稻瘟病": "推荐使用75%三环唑可湿性粉剂1000倍液预防，发病后用40%稻瘟灵乳油800倍液治疗。",
    "纹枯病": "建议使用5%井冈霉素水剂500倍液喷雾，注意田间水位管理，及时晒田。",
    "炭疽病": "推荐使用25%咪鲜胺乳油1000倍液或10%苯醚甲环唑水分散粒剂1500倍液喷雾。",
    "病毒病": "及时防治蚜虫等传毒媒介，使用20%病毒A可湿性粉剂500倍液，拔除病株。",
    "根腐病": "建议用50%多菌灵可湿性粉剂400倍液灌根，注意排水，避免连作。",
    "黑腐病": "使用72%农用链霉素可溶性粉剂4000倍液喷雾，清除病株，实行轮作。",
    "default": "建议加强田间管理，保持通风透光，合理施肥浇水。如症状持续，请咨询当地农技站。",
}


def get_crop_info(categories):
    """根据COCO类别获取作物信息"""
    for cat_name in categories:
        cat_lower = cat_name.lower()
        if cat_lower in CROP_CATEGORY_MAP:
            return CROP_CATEGORY_MAP[cat_lower]
    return GENERIC_CROP


def build_diagnosis_conversation(crop_info, img_info, template):
    """构建农业诊断对话"""
    crop_name = crop_info["crop"]
    crop_type = crop_info["type"]
    diseases = crop_info["common_diseases"]
    disease_name = random.choice(diseases)
    treatment = TREATMENT_ADVICE.get(disease_name, TREATMENT_ADVICE["default"])

    severity = random.choice(["轻度", "中度", "无明显症状"])

    if "无明显症状" in severity:
        diagnosis = (
            f"{template['assistant_prefix']}{crop_name}（{crop_type}）。\n"
            f"图片分辨率为{img_info['width']}x{img_info['height']}，"
            f"叶片颜色正常，暂未发现明显病斑或虫害迹象。\n"
            f"建议继续保持良好的田间管理，定期巡查，注意预防{diseases[0]}等常见病害。"
        )
    else:
        diagnosis = (
            f"{template['assistant_prefix']}{crop_name}（{crop_type}）。\n"
            f"图片分辨率为{img_info['width']}x{img_info['height']}，"
            f"观察到{severity}的异常症状，初步判断可能为{disease_name}。\n\n"
            f"诊断依据：叶片出现特征性病斑，符合{disease_name}的典型症状表现。\n\n"
            f"防治建议：{treatment}\n"
            f"同时建议加强田间通风，控制湿度，定期清理病残体。"
        )

    return diagnosis


def build_coco_conversations(coco, img_dir, max_samples=None):
    """将COCO标注转换为农业诊断对话格式"""
    img_ids = coco.getImgIds()
    if max_samples:
        img_ids = random.sample(img_ids, min(max_samples, len(img_ids)))

    conversations = []

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        categories = set()
        for ann in anns:
            cat = coco.loadCats(ann["category_id"])[0]
            categories.add(cat["name"])

        if not categories and not anns:
            continue

        crop_info = get_crop_info(categories)
        template = random.choice(INSTRUCTION_TEMPLATES)
        diagnosis = build_diagnosis_conversation(crop_info, img_info, template)

        conversation = {
            "images": [img_path],
            "messages": [
                {"role": "user", "content": template["user"]},
                {"role": "assistant", "content": diagnosis},
            ],
        }
        conversations.append(conversation)

    return conversations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="农业病虫害诊断数据预处理")
    parser.add_argument(
        "--coco-dir", default="./data/COCO2017", help="COCO2017数据集根目录"
    )
    parser.add_argument(
        "--output", default="./data/COCO2017/mllm_format_llava_instruct_data.json", help="输出JSON路径（须与 main.py 的 --data-dir 一致）"
    )
    parser.add_argument(
        "--ann-file",
        default="annotations/instances_train2017.json",
        help="COCO标注文件相对路径（使用instances而非captions以获取类别信息）",
    )
    parser.add_argument(
        "--img-dir", default="images/train2017", help="图片目录相对路径"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    args = parser.parse_args()

    random.seed(42)

    coco_dir = Path(args.coco_dir)
    ann_path = coco_dir / args.ann_file
    img_dir = coco_dir / args.img_dir

    print(f"加载COCO标注: {ann_path}")
    coco = COCO(str(ann_path))

    print(f"构建农业诊断对话数据 (图片目录: {img_dir})")
    conversations = build_coco_conversations(coco, str(img_dir), args.max_samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"完成: {output_path}")
    print(f"  总样本数: {len(conversations)}")
    print(f"  应用场景: 农业作物病虫害智能诊断")


if __name__ == "__main__":
    main()
