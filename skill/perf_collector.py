#!/usr/bin/env python3
"""
性能指标采集器
从 MindSpeed-MM 训练日志中提取 loss、grad_norm、elapsed_time 等指标。

解析口径（重要）
----------------
本工具**按 iteration 步号锚定**取数，而不是按日志中出现顺序追加：

    只在一行同时含 `iteration N/M` 与 `loss:` 时才收这个 loss。

原因：宽松正则 `loss[:\\s]+([0-9.]+)` 会把 `loss scale: 1024.0` 这类非迭代日志
也当成一步 loss 收进来。一旦某轮日志多打一行、少跑几步或被截断，后续所有步就
整体错位——图照出、数照算，但全是错的，且不报任何错。按步号锚定则两轮以步号
求交集对齐，对不上就明说，绝不错位。
"""
import re
import json
from pathlib import Path

RE_ITER = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
# 负向先行断言排除 loss_scale / loss_scale_value 等字段名
RE_LOSS = re.compile(r"(?<![_\w])loss\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?)")
RE_GRAD = re.compile(r"(?<![_\w])grad[_\s]*norm\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?)")
RE_TIME = re.compile(
    r"elapsed\s+time\s+per\s+iteration\s*\(\s*ms\s*\)\s*[:=]?\s*([0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)
RE_TIME_ALT = re.compile(r"elapsed\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE)
RE_GBS = re.compile(r"global\s+batch\s+size\s*[:=]\s*(\d+)", re.IGNORECASE)

# 统一评测口径: 取第 101-200 步均值（前 100 步为预热期）
# 口径来源: examples/qwen3_5/finetune_qwen3_5_0.8B.sh（注释自称"与上游示例逻辑一致"）
#   grep "elapsed time per iteration" ... | head -n 200 | tail -n 100
WINDOW_START, WINDOW_END = 101, 200


def parse_log(log_file):
    """按步号锚定解析训练日志，提取所有指标"""
    log_file = Path(log_file)
    if not log_file.exists():
        print(f"[ERROR] 日志文件不存在: {log_file}")
        return {}

    # step -> {"loss": .., "grad": .., "time": ..}
    recs = {}
    gbs = None
    dup = 0

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "iteration" not in line:
                continue
            mi = RE_ITER.search(line)
            if not mi:
                continue
            ml = RE_LOSS.search(line)
            if not ml:
                # 有 iteration 行但没解析出 loss（例如日志被截断）—— 该步视为缺失
                continue
            try:
                step = int(mi.group(1))
                loss = float(ml.group(1))
            except ValueError:
                continue

            grad = None
            mg = RE_GRAD.search(line)
            if mg:
                try:
                    grad = float(mg.group(1))
                except ValueError:
                    pass

            t = None
            mt = RE_TIME.search(line) or RE_TIME_ALT.search(line)
            if mt:
                try:
                    t = float(mt.group(1))
                except ValueError:
                    pass

            if step in recs:
                dup += 1
            recs[step] = {"loss": loss, "grad": grad, "time": t}

            if gbs is None:
                bg = RE_GBS.search(line)
                if bg:
                    gbs = int(bg.group(1))

    steps = sorted(recs)
    metrics = {
        "steps": steps,
        "loss_values": [recs[s]["loss"] for s in steps],
        "grad_norm_values": [recs[s]["grad"] for s in steps],
        "step_times_ms": [recs[s]["time"] for s in steps],
        "global_batch_size": gbs,
        "avg_step_time_ms": None,
        "samples_per_second": None,
        "duplicate_steps": dup,
    }
    metrics["total_steps"] = len(steps)

    # ---- 单步耗时统计（统一口径 101-200 步；步数不足则降级并明说）----
    win = [recs[s]["time"] for s in steps
           if WINDOW_START <= s <= WINDOW_END and recs[s]["time"] is not None]
    if win:
        metrics["eval_window"] = f"step {WINDOW_START}-{WINDOW_END}"
        eval_times = win
    else:
        allt = [r["time"] for r in recs.values() if r["time"] is not None]
        if allt:
            metrics["eval_window"] = (
                f"step 1-{metrics['total_steps']} (不足{WINDOW_END}步，降级统计；"
                f"性能轮 train_iters 需 >= {WINDOW_END})"
            )
        else:
            metrics["eval_window"] = "无耗时数据"
        eval_times = allt

    if eval_times:
        metrics["avg_step_time_ms"] = round(sum(eval_times) / len(eval_times), 1)
        metrics["eval_samples"] = len(eval_times)
    else:
        metrics["eval_samples"] = 0

    # 吞吐量 = GBS × 1000 / STEP_TIME(ms)
    if metrics["avg_step_time_ms"] and metrics["global_batch_size"]:
        metrics["samples_per_second"] = round(
            metrics["global_batch_size"] * 1000 / metrics["avg_step_time_ms"], 3)

    return metrics


def collect_all_metrics(log_file):
    """采集完整指标并生成结构化报告"""
    metrics = parse_log(log_file)
    if not metrics:
        return {"summary": {}, "accuracy_data": {},
                "steps": [], "loss_values": [], "grad_norm_values": [],
                "step_times_ms": []}

    loss_vals = metrics["loss_values"]
    grad_vals = metrics["grad_norm_values"]

    report = {
        "summary": {
            "total_steps": metrics["total_steps"],
            "avg_step_time_ms": metrics["avg_step_time_ms"],
            "global_batch_size": metrics["global_batch_size"],
            "samples_per_second": metrics["samples_per_second"],
            "eval_window": metrics.get("eval_window"),
            "eval_samples": metrics.get("eval_samples"),
        },
        # 关键点（第 50 / 100 步）—— 保留供人工核对，不作为判定依据
        "accuracy_data": {
            "step_50_loss": loss_vals[49] if len(loss_vals) >= 50 else None,
            "step_100_loss": loss_vals[99] if len(loss_vals) >= 100 else None,
            "step_50_grad_norm": grad_vals[49] if len(grad_vals) >= 50 else None,
            "step_100_grad_norm": grad_vals[99] if len(grad_vals) >= 100 else None,
        },
        "steps": metrics["steps"],
        "loss_values": loss_vals,
        "grad_norm_values": grad_vals,
        "step_times_ms": metrics["step_times_ms"],
    }

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python perf_collector.py <log_file>")
        sys.exit(1)

    result = collect_all_metrics(sys.argv[1])
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"\nloss数据点: {len(result['loss_values'])}")
    print(f"grad_norm数据点: {len(result['grad_norm_values'])}")
    print(f"step_time数据点: {len(result['step_times_ms'])}")
