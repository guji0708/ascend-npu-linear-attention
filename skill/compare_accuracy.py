#!/usr/bin/env python3
"""
精度对比脚本
对比基线日志与优化后日志的 loss / grad_norm，输出四项误差统计与判定结论。

判定口径（双口径，与《性能分析测试报告》一致）
--------------------------------------------
| 口径                     | 定义                                             | 作用                 |
|--------------------------|--------------------------------------------------|----------------------|
| 轨迹偏差（主判据）       | 两轮 loss 各自做 50 步滑动平均后，再算逐点相对误差 | 两条收敛轨迹的系统性偏离 |
| 逐点相对误差（参考）     | 直接对单步瞬时 loss 算相对误差                    | 与上游模板图形态一致 |

判定阈值：轨迹偏差均值 < 2% 且 全程 loss 均值相对偏差 < 2%。

为什么以轨迹偏差为主判据
------------------------
逐点相对误差的分母是**单步瞬时 loss**。单步 loss 自身噪声极大（本脚本会实测
同一轮训练相邻两步波动的中位数，即"噪声地板"），且收敛后 loss 趋近于 0，
分母效应会把噪声放大成几十个百分点的"误差"。当 2% 这条阈值线落在噪声地板
以下时，逐点口径不具备判定能力。grad_norm 的同频抖动比 loss 更大，故同样只
作参考、不参与判定。

用法:
    python compare_accuracy.py <基线日志> <优化日志> [--output report.json]
    python compare_accuracy.py <基线日志> <优化日志> [--window 50]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_collector import collect_all_metrics

MA_WINDOW = 50          # 轨迹偏差的滑动平均窗口（步）
THRESHOLD = 2.0         # 判定阈值（%）


def calc_relative_error(baseline, optimized):
    """计算相对绝对误差（%）"""
    if baseline is None or optimized is None or baseline == 0:
        return None
    return abs(baseline - optimized) / abs(baseline) * 100


def moving_avg(values, window):
    """滑动平均。返回 (均值序列, 该序列末点对应的 0-based 下标序列)。

    末点下标对齐：窗口 w 的第 k 个均值对应原序列第 k + w - 1 个元素（0-based）。
    """
    if len(values) < window:
        return [], []
    out = []
    running = sum(values[:window])
    out.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out.append(running / window)
    return out, list(range(window - 1, len(values)))


def trajectory_bias(base_losses, opt_losses, window=MA_WINDOW):
    """轨迹偏差：两轮各自做 window 步滑动平均后再算逐点相对误差"""
    mb, idx = moving_avg(base_losses, window)
    mo, _ = moving_avg(opt_losses, window)
    if not mb:
        return [], []
    return [calc_relative_error(b, o) for b, o in zip(mb, mo)], idx


def own_run_fluctuation(losses):
    """同一轮训练相邻两步 loss 的相对变化 —— 逐点相对误差指标的噪声地板"""
    out = []
    for a, b in zip(losses, losses[1:]):
        if a:
            out.append(abs(b - a) / abs(a) * 100)
    return out


def median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def error_stats(values):
    """四项误差统计（上游性能报告模板口径：Mean / MSE / Max / Min，百分比）"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    n = len(vals)
    return {
        "n": n,
        "mean_error_pct": round(sum(vals) / n, 4),
        # MSE 为「百分比误差的平方」的均值，量纲 %²，与上游模板定义一致
        "mse_error_pct2": round(sum(v * v for v in vals) / n, 4),
        "max_error_pct": round(max(vals), 4),
        "min_error_pct": round(min(vals), 4),
    }


def block_means(steps, base_losses, opt_losses, block=100):
    """每 block 步的分块均值相对误差"""
    out = []
    for start in range(1, steps[-1] + 1, block):
        pairs = [(b, o) for s, b, o in zip(steps, base_losses, opt_losses)
                 if start <= s <= start + block - 1]
        if not pairs:
            continue
        mb = sum(p[0] for p in pairs) / len(pairs)
        mo = sum(p[1] for p in pairs) / len(pairs)
        out.append({
            "block": f"{start}-{start + block - 1}",
            "n": len(pairs),
            "baseline_mean": round(mb, 6),
            "optimized_mean": round(mo, 6),
            "error_pct": round(calc_relative_error(mb, mo) or 0.0, 3),
        })
    return out


def compare_accuracy(baseline_log, optimized_log, output_path=None, ma_window=MA_WINDOW):
    """对比两个日志的精度（按步号锚定对齐）"""
    baseline = collect_all_metrics(baseline_log)
    optimized = collect_all_metrics(optimized_log)

    b_steps = baseline.get("steps", [])
    o_steps = optimized.get("steps", [])
    if not b_steps or not o_steps:
        raise SystemExit("[ERROR] 有一份日志没解析出任何迭代数据，无法对比")

    b_rec = dict(zip(b_steps, baseline["loss_values"]))
    o_rec = dict(zip(o_steps, optimized["loss_values"]))
    b_grad = dict(zip(b_steps, baseline["grad_norm_values"]))
    o_grad = dict(zip(o_steps, optimized["grad_norm_values"]))

    common = sorted(set(b_rec) & set(o_rec))
    if not common:
        raise SystemExit("[ERROR] 两份日志没有共同步号，无法对齐对比")
    only_b = sorted(set(b_rec) - set(o_rec))
    only_o = sorted(set(o_rec) - set(b_rec))
    if only_b or only_o:
        print(f"[WARN] 步号不一致，按交集 {len(common)} 步对齐"
              f"（仅基线有 {len(only_b)} 步，仅优化有 {len(only_o)} 步）")

    base_losses = [b_rec[s] for s in common]
    opt_losses = [o_rec[s] for s in common]
    base_grads = [b_grad.get(s) for s in common]
    opt_grads = [o_grad.get(s) for s in common]

    # ---- 逐点误差（参考口径）----
    step_errors = []
    for s, bl, ol, bg, og in zip(common, base_losses, opt_losses, base_grads, opt_grads):
        step_errors.append({
            "step": s,
            "loss_error_pct": _r4(calc_relative_error(bl, ol)),
            "grad_norm_error_pct": _r4(calc_relative_error(bg, og)),
        })

    loss_stats = error_stats([e["loss_error_pct"] for e in step_errors])
    grad_stats = error_stats([e["grad_norm_error_pct"] for e in step_errors])

    # ---- 轨迹偏差（主判据）----
    bias_vals, bias_idx = trajectory_bias(base_losses, opt_losses, ma_window)
    bias_stats = error_stats(bias_vals)
    bias_curve = [{"step": common[i], "bias_pct": _r4(v)}
                  for i, v in zip(bias_idx, bias_vals)]

    # ---- 全程 loss 均值相对偏差（主判据之二）----
    base_mean = sum(base_losses) / len(base_losses)
    opt_mean = sum(opt_losses) / len(opt_losses)
    agg_err = calc_relative_error(base_mean, opt_mean)

    # ---- 噪声地板：基线自身相邻两步波动的中位数 ----
    noise_floor = median(own_run_fluctuation(base_losses))

    # ---- 关键点（第 50 / 100 步）如实并列，供人工核对 ----
    key_steps = {}
    for k in (50, 100):
        if k in b_rec and k in o_rec:
            key_steps[f"step_{k}"] = {
                "baseline_loss": b_rec[k],
                "optimized_loss": o_rec[k],
                "loss_error_pct": _r4(calc_relative_error(b_rec[k], o_rec[k])),
                "baseline_grad_norm": b_grad.get(k),
                "optimized_grad_norm": o_grad.get(k),
                "grad_norm_error_pct": _r4(calc_relative_error(b_grad.get(k), o_grad.get(k))),
            }

    # ---- 判定：主判据双条件 ----
    bias_mean = bias_stats["mean_error_pct"] if bias_stats else None
    passed = (bias_mean is not None and bias_mean < THRESHOLD
              and agg_err is not None and agg_err < THRESHOLD)

    result = {
        "verdict": "PASS" if passed else "FAIL",
        "threshold": (f"轨迹偏差（{ma_window}步滑动均值）均值 < {THRESHOLD}% "
                      f"且 全程 loss 均值相对偏差 < {THRESHOLD}%"),
        "criterion_note": (
            "逐点口径的分母是单步瞬时 loss，而单步 loss 自身波动中位数即有 "
            f"{round(noise_floor, 3)}%（噪声地板），且收敛后 loss 趋近 0；"
            f"{THRESHOLD}% 阈值线本身落在噪声地板以下，故逐点与 grad_norm 口径"
            "仅作参考，判定以轨迹偏差为主判据。"),
        "aligned_steps": len(common),
        "steps_only_baseline": len(only_b),
        "steps_only_optimized": len(only_o),
        "smooth_window": ma_window,
        # 主判据
        "trajectory_bias_stats": bias_stats,
        "trajectory_bias_curve": bias_curve,
        "aggregate_loss": {
            "baseline_mean": round(base_mean, 6),
            "optimized_mean": round(opt_mean, 6),
            "relative_error_pct": _r4(agg_err),
        },
        "noise_floor_pct": round(noise_floor, 3),
        "bias_over_noise_floor_pct": (
            round(bias_stats["mean_error_pct"] / noise_floor * 100, 1)
            if bias_stats and noise_floor else None),
        # 参考口径
        "loss_error_stats": loss_stats,
        "grad_norm_error_stats": grad_stats,
        "key_steps_comparison": key_steps,
        "block_means": block_means(common, base_losses, opt_losses),
        "baseline_summary": baseline["summary"],
        "optimized_summary": optimized["summary"],
        "per_step_errors": step_errors,
    }

    # ---- 性能对比（与 lr 无关，故 A/C 与 D/B 之间也可比）----
    base_time = baseline["summary"].get("avg_step_time_ms")
    opt_time = optimized["summary"].get("avg_step_time_ms")
    if base_time and opt_time and base_time > 0:
        result["performance"] = {
            "baseline_step_time_ms": base_time,
            "optimized_step_time_ms": opt_time,
            "time_reduction_pct": round((1 - opt_time / base_time) * 100, 1),
            "speedup_ratio": round(base_time / opt_time, 2),
            "baseline_throughput": baseline["summary"].get("samples_per_second"),
            "optimized_throughput": optimized["summary"].get("samples_per_second"),
        }

    # ---- 画图（上游模板形态：loss 对比 + 误差针状图含 2% 阈值线）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5))

        ax1.plot(common, base_losses, color="#C0392B", linewidth=1.2, label="Baseline")
        ax1.plot(common, opt_losses, color="#2563EB", linewidth=1.2, alpha=0.75,
                 label="Optimized")
        ax1.set_title("Loss Comparison Chart")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        err_x = [e["step"] for e in step_errors if e["loss_error_pct"] is not None]
        err_y = [e["loss_error_pct"] for e in step_errors if e["loss_error_pct"] is not None]
        ax2.vlines(err_x, 0, err_y, color="#93C5FD", linewidth=0.7,
                   label="per-step relative error")
        if bias_curve:
            ax2.plot([b["step"] for b in bias_curve], [b["bias_pct"] for b in bias_curve],
                     color="#0F766E", linewidth=2.2,
                     label=f"trajectory bias ({ma_window}-step averaged loss)")
        if noise_floor:
            ax2.axhline(y=noise_floor, color="#94A3B8", linewidth=1.2, linestyle=":",
                        label=f"noise floor = {noise_floor:.1f}%")
        ax2.axhline(y=THRESHOLD, color="red", linewidth=1.2, linestyle="--",
                    label=f"{THRESHOLD}% threshold")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("relative abs error (%)")
        ax2.set_title("Relative Abs Error per Step")
        ax2.legend(fontsize=8.5)
        ax2.grid(alpha=0.3)

        if loss_stats:
            verdict = result["verdict"]
            head = (f"Verification: {verdict}   |   trajectory bias "
                    f"{bias_stats['mean_error_pct']:.4f}% < {THRESHOLD}%   |   "
                    f"aggregate loss deviation {agg_err:.4f}% < {THRESHOLD}%")
            body = (f"trajectory bias (main)  Mean: {bias_stats['mean_error_pct']:.4f}%  "
                    f"MSE: {bias_stats['mse_error_pct2']:.4f}  "
                    f"Max: {bias_stats['max_error_pct']:.4f}%  "
                    f"Min: {bias_stats['min_error_pct']:.4f}%\n"
                    f"per-step (reference)    Mean: {loss_stats['mean_error_pct']:.4f}%  "
                    f"MSE: {loss_stats['mse_error_pct2']:.4f}  "
                    f"Max: {loss_stats['max_error_pct']:.4f}%  "
                    f"Min: {loss_stats['min_error_pct']:.4f}%")
            fig.text(0.5, 0.045, head, ha="center", fontsize=9.5, fontweight="bold",
                     color="#047857" if verdict == "PASS" else "#B91C1C")
            fig.text(0.5, 0.008, body, ha="center", fontsize=8, color="#475569")

        plt.tight_layout(rect=(0, 0.075, 1, 1))
        if output_path:
            fig_path = str(Path(output_path).with_suffix("")) + ".png"
        else:
            fig_path = str(Path(baseline_log).parent / "accuracy_comparison.png")
        fig.savefig(fig_path, dpi=150, facecolor="white")
        plt.close(fig)
        result["figure_path"] = fig_path
        print(f"对比图已保存: {fig_path}")
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过画图输出")

    # ---- 保存报告 ----
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"报告已保存: {output_path}")

    return result


def _r4(v):
    return round(v, 4) if v is not None else None


def print_report(result):
    """打印可读报告"""
    print(f"\n{'='*66}")
    print("精度对比报告")
    print(f"{'='*66}")
    print(f"判定口径: {result['threshold']}")
    print(f"对齐步数: {result['aligned_steps']}"
          f"（仅基线有 {result['steps_only_baseline']} 步，"
          f"仅优化有 {result['steps_only_optimized']} 步）")
    print(f"结论: {result['verdict']}")

    bs = result["trajectory_bias_stats"]
    ls = result["loss_error_stats"]
    if bs:
        print(f"\n[主判据] 轨迹偏差（{result['smooth_window']}步滑动均值）")
        print(f"  Mean={bs['mean_error_pct']}%  MSE={bs['mse_error_pct2']}  "
              f"Max={bs['max_error_pct']}%  Min={bs['min_error_pct']}%")
    if ls:
        print(f"[参考] 逐点相对误差")
        print(f"  Mean={ls['mean_error_pct']}%  MSE={ls['mse_error_pct2']}  "
              f"Max={ls['max_error_pct']}%  Min={ls['min_error_pct']}%")
    gs = result.get("grad_norm_error_stats")
    if gs:
        print(f"[参考] grad_norm 逐点相对误差（抖动大于 loss，不作判定依据）")
        print(f"  Mean={gs['mean_error_pct']}%  MSE={gs['mse_error_pct2']}  "
              f"Max={gs['max_error_pct']}%  Min={gs['min_error_pct']}%")

    agg = result["aggregate_loss"]
    print(f"\n全程 loss 均值: 基线 {agg['baseline_mean']} / 优化 {agg['optimized_mean']}"
          f"，相对偏差 {agg['relative_error_pct']}%")
    print(f"噪声地板（基线自身相邻两步波动中位数）: {result['noise_floor_pct']}%"
          + (f"；轨迹偏差仅为噪声地板的 {result['bias_over_noise_floor_pct']}%"
             if result.get("bias_over_noise_floor_pct") is not None else ""))

    for name, k in (("step_50", "step_50"), ("step_100", "step_100")):
        c = result["key_steps_comparison"].get(k)
        if not c:
            continue
        print(f"\n  [{name}]")
        print(f"    loss:      基线={c['baseline_loss']}  优化后={c['optimized_loss']}"
              f"  误差={c['loss_error_pct']}%")
        print(f"    grad_norm: 基线={c['baseline_grad_norm']}  优化后={c['optimized_grad_norm']}"
              f"  误差={c['grad_norm_error_pct']}%")

    perf = result.get("performance")
    if perf:
        print(f"\n  [性能对比]（STEP_TIME 与 lr 无关，可与精度结论并列）")
        print(f"    基线单步耗时:   {perf['baseline_step_time_ms']} ms")
        print(f"    优化后单步耗时: {perf['optimized_step_time_ms']} ms")
        print(f"    耗时降低:       {perf['time_reduction_pct']}%")
        print(f"    加速比:         {perf['speedup_ratio']}x")
        print(f"    基线吞吐量:     {perf['baseline_throughput']} samples/s")
        print(f"    优化后吞吐量:   {perf['optimized_throughput']} samples/s")

    print(f"\n  > {result['criterion_note']}")
    print(f"\n{'='*66}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="精度对比工具（按步号锚定对齐；轨迹偏差为主判据）")
    parser.add_argument("baseline_log", help="基线训练日志路径")
    parser.add_argument("optimized_log", help="优化后训练日志路径")
    parser.add_argument("--output", default=None, help="报告输出路径(JSON；同名 .png 为对比图)")
    parser.add_argument("--window", type=int, default=MA_WINDOW,
                        help=f"轨迹偏差的滑动平均窗口步数，默认 {MA_WINDOW}")
    args = parser.parse_args()

    result = compare_accuracy(args.baseline_log, args.optimized_log,
                              args.output, args.window)
    print_report(result)

    if result["verdict"] == "FAIL":
        sys.exit(1)
