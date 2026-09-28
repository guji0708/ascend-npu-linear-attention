#!/usr/bin/env python3
"""
Qwen3.5-0.8B 昇腾NPU迁移Skill - 主入口
一键完成：模型下载 → 权重转换 → 配置生成 → 训练启动 → 性能采集

环境适配说明（详见《环境适配问题与解法》）：
  1. pyproject 的 requires-python 需放宽到 >=3.11,<3.13（镜像无 3.12，且不能换解释器）
  2. 依赖不能用 `pip install -e .`，需按名单补装（transformers 钉 4.57.0 与官方 5.2.0 冲突）
  3. 训练必须带 NON_MEGATRON=true（插件式 FSDP2 不走 Megatron 桥接）
  4. triton 必须先打补丁才能用：triton-ascend 的 npu_utils.cpp 引用了 CANN 9.0.0 不存在的
     RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE，打 ascend_porting/patch_triton.py 后可用
     （B 参考轮与 Conv1d 隔离实验即走此路径）；不打补丁才退回 eager
  5. gdn 为 eager 时 skip_gdn_recompute / skip_flash_attn_recompute 必须为 False
  6. 仓库根需有 ckpt / dataset 软链接（配置里是相对路径）
"""
import argparse
import importlib.util
import os
import sys
import subprocess
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent / "MindSpeed-MM"

# 精度对齐轮步数（复现参考精度日志）
ACCURACY_ITERS = 100
# 性能轮步数：上游示例脚本取第 101-200 步均值，故必须 >= 200
PERF_ITERS = 200
# 参考精度日志对应的数据规模
OFFICIAL_MAX_SAMPLES = 1000
OFFICIAL_GRAD_ACCUM = 8


def run(cmd, cwd=None, check=True):
    print(f"\n{'='*60}")
    print(f"[RUN] {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def detect_gdn_implementation():
    """自动选择 GDN 算子实现。

    ascendc 性能最佳，但需要编译安装 fla_npu 算子库；
    triton 可用，但必须先打 ascend_porting/patch_triton.py 补丁（triton-ascend 的
    npu_utils.cpp 引用了 CANN 9.0.0 不存在的 RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE）；
    探测不到 fla_npu 时默认退回 eager —— 那是零额外依赖的基线档，不是唯一可选项。
    """
    try:
        return "ascendc" if importlib.util.find_spec("fla_npu") is not None else "eager"
    except Exception:
        return "eager"


def step1_download_model(model_dir):
    """从ModelScope下载Qwen3.5-0.8B模型权重"""
    print("\n[Step 1] 下载Qwen3.5-0.8B模型权重")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if (model_dir / "config.json").exists():
        print(f"  模型已存在: {model_dir}")
        return

    run(
        f"modelscope download --model Qwen/Qwen3.5-0.8B "
        f"--local_dir {model_dir}",
        check=True,
    )
    print(f"  下载完成: {model_dir}")


def step2_convert_weight(hf_dir, dcp_dir, num_workers=4):
    """将HF权重转换为MindSpeed-MM的DCP格式"""
    print("\n[Step 2] 权重格式转换 HF → DCP")
    hf_dir = Path(hf_dir)
    dcp_dir = Path(dcp_dir)

    if dcp_dir.exists() and any(dcp_dir.iterdir()):
        print(f"  DCP权重已存在: {dcp_dir}")
        return

    dcp_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"mm-convert Qwen35Converter hf_to_dcp "
        f"--hf_dir {hf_dir} "
        f"--dcp_dir {dcp_dir} "
        f"--num_workers {num_workers}",
        cwd=str(REPO_DIR),
        check=True,
    )
    print(f"  转换完成: {dcp_dir}")


def ensure_repo_links():
    """确保仓库根有 ckpt / dataset 软链接。

    配置里用的是相对路径（./ckpt/...、./dataset/...），而训练时工作目录是仓库根，
    因此权重和数据集必须在仓库内可见。ckpt 常已存在一个空目录，须先删掉才能建链接。
    """
    print("\n[Step 3.0] 检查仓库内 ckpt / dataset 软链接")
    for name in ("ckpt", "dataset"):
        link = REPO_DIR / name
        target = BASE_DIR.parent / name
        if link.is_symlink():
            print(f"  {name} 已是软链接 -> {os.readlink(link)}")
            continue
        if link.is_dir():
            if any(link.iterdir()):
                print(f"  [WARN] {name} 是非空真实目录，跳过（请手动确认）")
                continue
            link.rmdir()
            print(f"  已删除空目录 {name}")
        if target.exists():
            link.symlink_to(target, target_is_directory=True)
            print(f"  已建软链接 {name} -> {target}")
        else:
            print(f"  [WARN] 目标不存在: {target}，跳过")


def step3_generate_config(hf_dir, dcp_dir, data_dir, output_path, npus=1,
                          gdn=None, train_iters=None):
    """生成Qwen3.5-0.8B训练配置文件

    从上游配置模板派生，并覆盖全部经验证的环境适配项与路径。
    """
    print("\n[Step 3] 生成训练配置文件")
    import yaml

    output_path = Path(output_path).resolve()
    # 上游 master 仓库只有 4B 模板，0.8B 配置由它派生（字段结构完全同构）
    template_path = REPO_DIR / "examples" / "qwen3_5" / "qwen3_5_0.8B_config.yaml"
    if not template_path.exists():
        template_path = REPO_DIR / "examples" / "qwen3_5" / "qwen3_5_4B_config.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"找不到配置模板: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- 路径覆盖 ----
    hf_path = str(hf_dir) + "/"
    config["data"]["dataset_param"]["preprocess_parameters"]["model_name_or_path"] = hf_path
    config["model"]["model_name_or_path"] = hf_path
    config["training"]["load"] = str(dcp_dir)

    data_dir = Path(data_dir).resolve()
    dataset_json = None
    for cand in ("annotations_slim.json", "mllm_format_llava_instruct_data.json"):
        p = data_dir / cand
        if p.exists():
            dataset_json = p
            break
    if dataset_json is None:
        raise FileNotFoundError(
            f"在 {data_dir} 中未找到 annotations_slim.json 或 "
            f"mllm_format_llava_instruct_data.json，请先准备数据集并解压")
    config["data"]["dataset_param"]["basic_parameters"]["dataset_dir"] = str(data_dir) + "/"
    config["data"]["dataset_param"]["basic_parameters"]["dataset"] = str(dataset_json)
    print(f"  数据集标注: {dataset_json}")
    print(f"  数据集根目录: {data_dir}/")

    # ---- 环境适配项（踩坑经验，勿删）----
    # (1) GDN / causal_conv1d 算子实现
    gdn_impl = gdn or detect_gdn_implementation()
    config["model"]["gdn_implementation"] = gdn_impl
    # gdn=ascendc 时 causal_conv1d 只支持 triton/ascendc（避免算子间布局不匹配）
    config["model"]["causal_conv1d_implementation"] = (
        "triton" if gdn_impl == "ascendc" else "eager")
    if gdn_impl == "ascendc":
        print("  gdn_implementation = ascendc（已检测到 fla_npu）")
    else:
        print("  gdn_implementation = eager"
              "（未检测到 fla_npu，默认退回零依赖的 eager 基线档；"
              "若要跑 triton 需先打 ascend_porting/patch_triton.py）")

    # (2) eager 与 skip_*_recompute 互斥，否则 overwrite_transformer_config 直接抛 ValueError
    if gdn_impl != "ascendc":
        config["model"]["skip_gdn_recompute"] = False
        config["model"]["skip_flash_attn_recompute"] = False
        print("  skip_gdn_recompute / skip_flash_attn_recompute = False（与 eager 配套）")

    # (3) 并行与训练规模
    config["parallel"]["fully_shard_parallel_size"] = npus
    config["training"]["gradient_accumulation_steps"] = OFFICIAL_GRAD_ACCUM
    config["training"]["train_iters"] = train_iters or ACCURACY_ITERS
    config["data"]["dataset_param"]["basic_parameters"]["max_samples"] = OFFICIAL_MAX_SAMPLES

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  配置已生成: {output_path}")
    print(f"  train_iters = {config['training']['train_iters']}"
          f"（性能轮需 >= {PERF_ITERS}，上游示例取第 101-{PERF_ITERS} 步）")
    return str(output_path)


def step4_start_training(config_path, npus=1, log_dir=None):
    """启动NPU训练"""
    print("\n[Step 4] 启动昇腾NPU训练")
    log_dir = Path(log_dir) if log_dir else REPO_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    env_exports = (
        "source /usr/local/Ascend/cann/set_env.sh && "
        "export NON_MEGATRON=true && "
        "export MULTI_STREAM_MEMORY_REUSE=2 && "
        "export TASK_QUEUE_ENABLE=2 && "
        "export ASCEND_LAUNCH_BLOCKING=0 && "
        "export ACLNN_CACHE_LIMIT=100000 && "
        "export CPU_AFFINITY_CONF=1 && "
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
    )

    torchrun_cmd = (
        f"torchrun --nproc_per_node {npus} "
        f"--nnodes 1 --node_rank 0 "
        f"--master_addr localhost --master_port 6000 "
        f"mindspeed_mm/fsdp/train/trainer.py {config_path}"
    )

    full_cmd = f'{env_exports} && {torchrun_cmd} 2>&1 | tee {log_file}'
    print(f"  日志文件: {log_file}")
    run(full_cmd, cwd=str(REPO_DIR), check=False)

    return str(log_file)


def step5_collect_metrics(log_file):
    """从训练日志中采集性能和精度指标"""
    print("\n[Step 5] 采集性能与精度指标")
    from perf_collector import collect_all_metrics

    metrics = collect_all_metrics(log_file)

    report_path = REPO_DIR.parent / "docs" / f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  指标报告已保存: {report_path}")
    print(f"  - 统计窗口: {metrics.get('eval_window', 'N/A')}")
    print(f"  - 窗口内样本数: {metrics.get('eval_samples', 'N/A')}")
    print(f"  - loss数据点: {len(metrics.get('loss_values', []))}")
    print(f"  - grad_norm数据点: {len(metrics.get('grad_norm_values', []))}")
    print(f"  - 平均单步耗时: {metrics.get('avg_step_time_ms', 'N/A')} ms")
    print(f"  - 吞吐量: {metrics.get('samples_per_second', 'N/A')} samples/s")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-0.8B 昇腾NPU迁移Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 精度对齐轮（100 步）
  python main.py --all --npus 1 --data-dir /workspace/ascend_ws/dataset \\
      --hf-dir /workspace/ascend_ws/ckpt/hf_path/Qwen3.5-0.8B \\
      --dcp-dir /workspace/ascend_ws/ckpt/dcp_path/Qwen3.5-0.8B

  # 性能轮（200 步，上游示例取第 101-200 步均值）
  python main.py --train --train-iters 200 --npus 1

  # 仅下载模型
  python main.py --download

  # 仅转换权重
  python main.py --convert --hf-dir ./ckpt/hf_path/Qwen3.5-0.8B

  # 仅采集指标
  python main.py --collect --log logs/train_20260921_204500.log
        """,
    )
    parser.add_argument("--all", action="store_true", help="执行完整流程")
    parser.add_argument("--download", action="store_true", help="Step1: 下载模型")
    parser.add_argument("--convert", action="store_true", help="Step2: 权重转换")
    parser.add_argument("--config-gen", action="store_true", help="Step3: 生成配置")
    parser.add_argument("--train", action="store_true", help="Step4: 启动训练")
    parser.add_argument("--collect", action="store_true", help="Step5: 采集指标")

    parser.add_argument("--hf-dir", default="./ckpt/hf_path/Qwen3.5-0.8B", help="HF权重目录")
    parser.add_argument("--dcp-dir", default="./ckpt/dcp_path/Qwen3.5-0.8B", help="DCP权重目录")
    parser.add_argument("--data-dir", default="./data/COCO2017", help="数据集目录")
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "generated" / "qwen3_5_0.8B_config.yaml"),
        help="配置文件输出路径（config-gen时）/ 训练加载路径（train时）",
    )
    parser.add_argument("--npus", type=int, default=1, help="NPU卡数")
    parser.add_argument("--train-iters", type=int, default=None,
                        help=f"训练步数，默认 {ACCURACY_ITERS}（精度对齐轮）；"
                             f"性能轮传 {PERF_ITERS}")
    parser.add_argument("--gdn", choices=["ascendc", "triton", "eager"], default=None,
                        help="GDN算子实现，默认自动探测 fla_npu（有则 ascendc，否则 eager）")
    parser.add_argument("--log", default=None, help="训练日志文件路径（采集指标用）")
    parser.add_argument("--workers", type=int, default=4, help="权重转换并行数")

    args = parser.parse_args()

    if not any([args.all, args.download, args.convert, args.config_gen,
                args.train, args.collect]):
        parser.print_help()
        return

    start_time = time.time()

    if args.all or args.download:
        step1_download_model(args.hf_dir)

    if args.all or args.convert:
        step2_convert_weight(args.hf_dir, args.dcp_dir, args.workers)

    if args.all:
        ensure_repo_links()

    if args.all or args.config_gen:
        config_path = step3_generate_config(
            args.hf_dir, args.dcp_dir, args.data_dir, args.config, args.npus,
            gdn=args.gdn, train_iters=args.train_iters,
        )
    else:
        config_path = args.config

    if args.all or args.train:
        log_file = step4_start_training(config_path, args.npus)
    else:
        log_file = args.log

    if args.all or args.collect:
        if log_file:
            step5_collect_metrics(log_file)
        else:
            print("[WARN] 未指定日志文件，跳过指标采集")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"全流程完成，总耗时: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
