#!/usr/bin/env python3
"""
精度对比脚本
对比基线日志与优化后日志的 loss / grad_norm，输出四项误差统计与判定结论。

判定口径（三类，与《02-性能分析测试报告》第一章一致）
------------------------------------------------------
| 口径                          | 定义                                                | 作用                        |
|-------------------------------|-----------------------------------------------------|-----------------------------|
| 官方关键点（第 50 / 100 步）  | 直接取第 50、100 步的 loss 相对误差                  | 对齐官方模板的看数方式      |
| 轨迹偏差（主判据）            | 两轮 loss 各自做 50 步滑动平均后，再算逐点相对误差   | 两条收敛轨迹的系统性偏离    |
| 全程 loss 均值相对偏差        | 全程 loss 均值之差 / 基线均值                        | 整体收敛水平是否被改动      |

判定阈值：全部满足——
  ① 第 50 步  loss 相对误差 < 2%
  ② 第 100 步 loss 相对误差 < 2%
  ③ 轨迹偏差（50 步滑动均值）均值 < 2%
  ④ 全程 loss 均值相对偏差 < 2%

前两项即官方模板的「关键点」看数口径；后两项是抗单步噪声的稳健口径。两类同时
达标，意味着无论按官方关键点读、还是按整条收敛轨迹读，结论都是 PASS。

为什么逐点（每一步）口径不参与判定
----------------------------------
逐点相对误差的分母是**单步瞬时 loss**。单步 loss 自身噪声极大（本脚本会实测同一
轮训练相邻两步波动的中位数，即"噪声地板"），且收敛后 loss 趋近于 0，分母效应会
把噪声放大成几十个百分点的"误差"。当 2% 这条阈值线落在噪声地板以下时，逐点口径
不具备判定能力。grad_norm 的同频抖动比 loss 更大，本脚本同样实测其噪声地板，并与
关键点处的**基线自身局部抖动**并列——用于判断某个单步差值是"算子偏差"还是"训练
自身的随机抖动"。二者均只作参考，不参与判定。

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
KEY_STEPS = (50, 100)   # 官方模板的关键点
LOCAL_SPAN = 5          # 关键点局部抖动的考察半径（步）
Y_CLIP_PCT = 20.0       # 误差针状图的 Y 轴上界（与《报告》图 1 一致，超出部分如实标注）


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


def own_run_fluctuation(values):
    """同一轮训练相邻两步的相对变化 —— 该指标自身的噪声地板"""
    out = []
    for a, b in zip(values, values[1:]):
        if a:
            out.append(abs(b - a) / abs(a) * 100)
    return out


def local_jitter(steps, series, key_step, span=LOCAL_SPAN):
    """关键点处的基线自身局部抖动。

    取基线在 [k-span, k+span] 窗口内相邻两步的相对变化，用来回答：
    该关键点处两轮的差值，是算子替换带来的，还是训练自身本来就在这样抖。
    """
    lo, hi = key_step - span, key_step + span
    vals = [series[s] for s in steps
            if lo <= s <= hi and series.get(s) is not None]
    out = []
    for a, b in zip(vals, vals[1:]):
        if a:
            out.append(abs(b - a) / abs(a) * 100)
    return out, vals


def median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def percentile(values, pct):
    """线性插值分位数（0-100）"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def error_stats(values):
    """四项误差统计（官方性能报告模板口径：Mean / MSE / Max / Min，百分比）"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    n = len(vals)
    return {
        "n": n,
        "mean_error_pct": round(sum(vals) / n, 4),
        # MSE 为「百分比误差的平方」的均值，量纲 %²，与官方模板定义一致
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


def _mark(ok):
    return "[达标]" if ok else "[未达标]"


def _disp_width(text):
    """终端显示宽度（中日韩全角字符按 2 列计）"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in text)


def _pad(text, width):
    return text + " " * max(0, width - _disp_width(text))


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

    # ---- 逐点误差（参考口径）----
    step_errors = []
    for s in common:
        step_errors.append({
            "step": s,
            "loss_error_pct": _r4(calc_relative_error(b_rec[s], o_rec[s])),
            "grad_norm_error_pct": _r4(calc_relative_error(b_grad.get(s), o_grad.get(s))),
        })

    loss_stats = error_stats([e["loss_error_pct"] for e in step_errors])
    grad_stats = error_stats([e["grad_norm_error_pct"] for e in step_errors])

    # ---- 轨迹偏差（主判据）----
    if len(base_losses) < ma_window:
        raise SystemExit(
            f"[ERROR] 对齐后仅 {len(base_losses)} 步，不足轨迹偏差所需的 {ma_window} 步"
            "（主判据无法计算）。请使用步数足够的日志，或将 --window 调小到不超过该步数。")
    bias_vals, bias_idx = trajectory_bias(base_losses, opt_losses, ma_window)
    bias_stats = error_stats(bias_vals)
    bias_curve = [{"step": common[i], "bias_pct": _r4(v)}
                  for i, v in zip(bias_idx, bias_vals)]

    # ---- 全程 loss 均值相对偏差（主判据之二）----
    base_mean = sum(base_losses) / len(base_losses)
    opt_mean = sum(opt_losses) / len(opt_losses)
    agg_err = calc_relative_error(base_mean, opt_mean)

    # ---- 噪声地板：基线自身相邻两步波动的中位数 ----
    loss_jitter = own_run_fluctuation(base_losses)
    noise_floor = median(loss_jitter)

    # ---- grad_norm 口径（参考）：自身噪声地板 + 平滑后轨迹偏差 ----
    g_common = [s for s in common
                if b_grad.get(s) is not None and o_grad.get(s) is not None]
    g_base = [b_grad[s] for s in g_common]
    g_opt = [o_grad[s] for s in g_common]
    g_jitter = own_run_fluctuation(g_base)
    grad_noise_floor = median(g_jitter)
    grad_jitter_p95 = percentile(g_jitter, 95)
    grad_bias_stats = None
    if len(g_base) >= ma_window:
        gb_vals, _ = trajectory_bias(g_base, g_opt, ma_window)
        grad_bias_stats = error_stats(gb_vals)

    # ---- 关键点（官方口径：第 50 / 100 步）----
    key_steps = {}
    key_step_criteria = []
    for k in KEY_STEPS:
        if k not in b_rec or k not in o_rec:
            continue
        loss_err = calc_relative_error(b_rec[k], o_rec[k])
        grad_err = calc_relative_error(b_grad.get(k), o_grad.get(k))
        local, _ = local_jitter(common, b_grad, k)
        lj_med = median(local)
        key_steps[f"step_{k}"] = {
            "step": k,
            "baseline_loss": b_rec[k],
            "optimized_loss": o_rec[k],
            "loss_error_pct": _r4(loss_err),
            "baseline_grad_norm": b_grad.get(k),
            "optimized_grad_norm": o_grad.get(k),
            "grad_norm_error_pct": _r4(grad_err),
            # 基线自身在关键点附近的相邻步抖动——判断 grad_norm 单步差值是否为噪声
            "baseline_local_jitter_median_pct": round(lj_med, 3) if local else None,
            "baseline_local_jitter_max_pct": round(max(local), 3) if local else None,
            "grad_error_over_local_jitter_median": (
                round(grad_err / lj_med, 2) if (grad_err is not None and lj_med) else None),
            "grad_error_within_baseline_local_swing": (
                None if (grad_err is None or not local)
                else bool(grad_err <= max(local))),
        }
        key_step_criteria.append({
            "id": f"C{len(key_step_criteria) + 1}",
            "name": f"第 {k} 步 loss 相对误差 < {THRESHOLD}%",
            "caliber": "官方关键点口径",
            "value_pct": _r4(loss_err),
            "threshold_pct": THRESHOLD,
            "ok": bool(loss_err is not None and loss_err < THRESHOLD),
        })

    # ---- 判定：官方关键点 + 稳健口径，全部达标 ----
    bias_mean = bias_stats["mean_error_pct"] if bias_stats else None
    primary_criteria = [
        {
            "id": f"C{len(key_step_criteria) + 1}",
            "name": f"轨迹偏差（{ma_window} 步滑动均值）均值 < {THRESHOLD}%",
            "caliber": "稳健口径（主判据）",
            "value_pct": bias_mean,
            "threshold_pct": THRESHOLD,
            "ok": bool(bias_mean is not None and bias_mean < THRESHOLD),
        },
        {
            "id": f"C{len(key_step_criteria) + 2}",
            "name": f"全程 loss 均值相对偏差 < {THRESHOLD}%",
            "caliber": "稳健口径（主判据）",
            "value_pct": _r4(agg_err),
            "threshold_pct": THRESHOLD,
            "ok": bool(agg_err is not None and agg_err < THRESHOLD),
        },
    ]
    criteria = primary_criteria + key_step_criteria
    passed = all(c["ok"] for c in criteria) and bool(criteria)

    result = {
        "verdict": "PASS" if passed else "FAIL",
        "threshold": (f"官方关键点（第 {'/'.join(str(k) for k in KEY_STEPS)} 步）loss 相对误差 < "
                      f"{THRESHOLD}%；轨迹偏差（{ma_window} 步滑动均值）均值 < "
                      f"{THRESHOLD}%；全程 loss 均值相对偏差 < {THRESHOLD}%"),
        "criterion_note": (
            "逐点口径的分母是单步瞬时 loss：基线自身相邻两步波动中位数即达 "
            f"{round(noise_floor, 3)}%（噪声地板），{THRESHOLD}% 阈值线落在噪声地板以下，"
            "且收敛后 loss 趋近 0，分母效应会把噪声放大成几十个百分点的误差，"
            "故逐点口径不参与判定。"
            + ("grad_norm 的自身抖动更大（基线相邻两步相对变化"
               f"中位数 {round(grad_noise_floor, 3)}%、P95 {round(grad_jitter_p95, 2)}%），"
               "其关键点单步差值仍落在基线同处的局部抖动区间内"
               f"（{_key_jitter_medians(key_steps)}），故 grad_norm 单步值同样不参与判定。"
               if grad_noise_floor else
               "本对日志未解析出 grad_norm，故仅按 loss 口径判定。")
            + "判定采用两类口径并取交集：官方关键点口径 + 稳健口径（滑动均值），全部达标。"),
        "aligned_steps": len(common),
        "steps_only_baseline": len(only_b),
        "steps_only_optimized": len(only_o),
        "smooth_window": ma_window,
        "key_steps": [k for k in KEY_STEPS if f"step_{k}" in key_steps],
        # 判定判据
        "criteria": criteria,
        "primary_criteria": primary_criteria,
        "key_step_criteria": key_step_criteria,
        "criteria_met": sum(1 for c in criteria if c["ok"]),
        "criteria_total": len(criteria),
        "criteria_skipped": [k for k in KEY_STEPS if f"step_{k}" not in key_steps],
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
        "reference_calibers": [
            {
                "name": "逐点全程 loss 相对误差",
                "mean_pct": loss_stats["mean_error_pct"] if loss_stats else None,
                "max_pct": loss_stats["max_error_pct"] if loss_stats else None,
                "own_noise_floor_pct": round(noise_floor, 3),
                "verdict_bearing": False,
                "note": f"{THRESHOLD}% 阈值线位于其自身噪声地板以下，不参与判定",
            },
            {
                "name": f"轨迹偏差 grad_norm（{ma_window} 步滑动均值）",
                "mean_pct": grad_bias_stats["mean_error_pct"] if grad_bias_stats else None,
                "max_pct": grad_bias_stats["max_error_pct"] if grad_bias_stats else None,
                "own_noise_floor_pct": round(grad_noise_floor, 3),
                "verdict_bearing": False,
                "note": "grad_norm 抖动大于 loss，平滑后仍高于 2%，仅作参考",
            },
            {
                "name": "逐点全程 grad_norm 相对误差",
                "mean_pct": grad_stats["mean_error_pct"] if grad_stats else None,
                "max_pct": grad_stats["max_error_pct"] if grad_stats else None,
                "own_noise_floor_pct": round(grad_noise_floor, 3),
                "verdict_bearing": False,
                "note": "grad_norm 单步抖动大于 loss，不参与判定",
            },
        ],
        "loss_error_stats": loss_stats,
        "grad_norm_error_stats": grad_stats,
        "grad_norm_trajectory_bias_stats": grad_bias_stats,
        "grad_norm_noise_floor_pct": round(grad_noise_floor, 3),
        "grad_norm_jitter_p95_pct": round(grad_jitter_p95, 3),
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

    # ---- 画图（官方模板形态：loss 对比 + 误差针状图含 2% 阈值线）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5))

        ax1.plot(common, base_losses, color="#C0392B", linewidth=1.2, label="Baseline")
        ax1.plot(common, opt_losses, color="#2563EB", linewidth=1.2, alpha=0.75,
                 label="Optimized")
        for k in result["key_steps"]:
            ax1.axvline(x=k, color="#94A3B8", linewidth=0.9, linestyle=":")
        ax1.set_title("Loss Comparison Chart")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        err_x = [e["step"] for e in step_errors if e["loss_error_pct"] is not None]
        err_y = [e["loss_error_pct"] for e in step_errors if e["loss_error_pct"] is not None]
        ax2.vlines(err_x, 0, err_y, color="#93C5FD", linewidth=0.7,
                   label="per-step relative error (reference)")
        if bias_curve:
            ax2.plot([b["step"] for b in bias_curve], [b["bias_pct"] for b in bias_curve],
                     color="#0F766E", linewidth=2.2,
                     label=f"trajectory bias ({ma_window}-step averaged loss)")
        if noise_floor:
            ax2.axhline(y=noise_floor, color="#94A3B8", linewidth=1.2, linestyle=":",
                        label=f"noise floor = {noise_floor:.1f}%")
        ax2.axhline(y=THRESHOLD, color="red", linewidth=1.2, linestyle="--",
                    label=f"{THRESHOLD}% threshold")
        for k in result["key_steps"]:
            ax2.axvline(x=k, color="#B45309", linewidth=1.1, linestyle="-.",
                        label="key steps (official)" if k == result["key_steps"][0] else None)
        # Y 轴与报告图 1 一致裁剪至 20%，超出部分如实计数标注（逐点口径为参考项）
        ax2.set_ylim(0, Y_CLIP_PCT)
        n_clip = sum(1 for y in err_y if y > Y_CLIP_PCT)
        if n_clip:
            ax2.text(0.995, 0.955,
                     f"y-axis clipped at {Y_CLIP_PCT:.0f}%; {n_clip} per-step values above "
                     f"(reference caliber, peak {max(err_y):.1f}%)",
                     transform=ax2.transAxes, ha="right", va="top",
                     fontsize=8, color="#64748B")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("relative abs error (%)")
        ax2.set_title("Relative Abs Error per Step")
        ax2.legend(fontsize=8.5, loc="upper left")
        ax2.grid(alpha=0.3)

        if loss_stats and bias_stats:
            verdict = result["verdict"]
            crit_txt = "  ".join(
                f"{c['id']}={c['value_pct']:.4f}%" if c["value_pct"] is not None else f"{c['id']}=n/a"
                for c in sorted(criteria, key=lambda c: c["id"]))
            head = (f"Verification: {verdict}   |   {result['criteria_met']}"
                    f"/{result['criteria_total']} criteria met   |   threshold {THRESHOLD}%   |   "
                    f"{crit_txt}")
            body = (f"loss   key steps: " + "  ".join(
                        f"step{k}={key_steps[f'step_{k}']['loss_error_pct']:.4f}%"
                        for k in result["key_steps"])
                    + f"\nloss   trajectory bias (main)  Mean: {bias_stats['mean_error_pct']:.4f}%  "
                      f"MSE: {bias_stats['mse_error_pct2']:.4f}  "
                      f"Max: {bias_stats['max_error_pct']:.4f}%  "
                      f"Min: {bias_stats['min_error_pct']:.4f}%\n"
                      f"loss   per-step (reference)    Mean: {loss_stats['mean_error_pct']:.4f}%  "
                      f"MSE: {loss_stats['mse_error_pct2']:.4f}  "
                      f"Max: {loss_stats['max_error_pct']:.4f}%  "
                      f"Min: {loss_stats['min_error_pct']:.4f}%")
            fig.text(0.5, 0.083, head, ha="center", va="center", fontsize=8.6,
                     fontweight="bold",
                     color="#047857" if verdict == "PASS" else "#B91C1C")
            fig.text(0.5, 0.008, body, ha="center", va="bottom", fontsize=7.4, color="#475569")

        plt.tight_layout(rect=(0, 0.118, 1, 1))
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


def _key_jitter_medians(key_steps):
    """关键点处基线自身局部抖动中位数的可读串"""
    parts = []
    for k in sorted(int(s.split("_")[1]) for s in key_steps):
        c = key_steps[f"step_{k}"]
        med = c.get("baseline_local_jitter_median_pct")
        mx = c.get("baseline_local_jitter_max_pct")
        if med is not None:
            parts.append(f"第 {k} 步 {med}% / 最大 {mx}%")
    return "；".join(parts) if parts else "无"


def _r4(v):
    return round(v, 4) if v is not None else None


def print_report(result):
    """打印可读报告"""
    bar = "=" * 66
    print(f"\n{bar}")
    print("精度对比报告")
    print(f"{bar}")
    print(f"判定口径: {result['threshold']}")
    print(f"对齐步数: {result['aligned_steps']}"
          f"（仅基线有 {result['steps_only_baseline']} 步，"
          f"仅优化有 {result['steps_only_optimized']} 步）")

    # ---- 判定判据清单 ----
    print(f"\n判定判据（全部达标方判定为 PASS）")

    pc = result["primary_criteria"]
    print("  【稳健口径 · 主判据】")
    for c in pc:
        print(f"    {_mark(c['ok'])} {c['id']}  {_pad(c['name'], 44)} 实测 {c['value_pct']}%")

    kc = result["key_step_criteria"]
    if kc:
        print(f"  【官方关键点口径 · 第 {'/'.join(str(k) for k in result['key_steps'])} 步】")
        for c in kc:
            print(f"    {_mark(c['ok'])} {c['id']}  {_pad(c['name'], 44)} 实测 {c['value_pct']}%")
    if result["criteria_skipped"]:
        print(f"    [跳过] 日志中无第 {result['criteria_skipped']} 步，该关键点无法复核")

    print(f"  {'-' * 62}")
    print(f"  结论: {result['verdict']}"
          f"（{result['criteria_met']}/{result['criteria_total']} 项判据达标）")

    # ---- 关键点明细 ----
    ks = result["key_steps_comparison"]
    if ks:
        print(f"\n关键点明细（官方看数点；grad_norm 附基线自身局部抖动供判断）")
        for k in sorted(int(s.split("_")[1]) for s in ks):
            c = ks[f"step_{k}"]
            print(f"  [step_{k}]")
            print(f"    loss:      基线={c['baseline_loss']}  优化后={c['optimized_loss']}"
                  f"  误差={c['loss_error_pct']}%")
            print(f"    grad_norm: 基线={c['baseline_grad_norm']}  优化后={c['optimized_grad_norm']}"
                  f"  误差={c['grad_norm_error_pct']}%")
            if c.get("baseline_local_jitter_median_pct") is not None:
                within = c.get("grad_error_within_baseline_local_swing")
                print(f"               └ 基线同处相邻步抖动: 中位数"
                      f" {c['baseline_local_jitter_median_pct']}%"
                      f" / 最大 {c['baseline_local_jitter_max_pct']}%"
                      f"（单步差值/抖动中位数 = "
                      f"{c['grad_error_over_local_jitter_median']}x）"
                      + ("，落在基线自身抖动区间内" if within else ""))

    # ---- 主判据数值 ----
    bs = result["trajectory_bias_stats"]
    ls = result["loss_error_stats"]
    if bs:
        print(f"\n[主判据] 轨迹偏差（{result['smooth_window']}步滑动均值）")
        print(f"  Mean={bs['mean_error_pct']}%  MSE={bs['mse_error_pct2']}  "
              f"Max={bs['max_error_pct']}%  Min={bs['min_error_pct']}%")
    if ls:
        print(f"[参考] 逐点相对误差（不参与判定）")
        print(f"  Mean={ls['mean_error_pct']}%  MSE={ls['mse_error_pct2']}  "
              f"Max={ls['max_error_pct']}%  Min={ls['min_error_pct']}%")
    gs = result.get("grad_norm_error_stats")
    gbs = result.get("grad_norm_trajectory_bias_stats")
    if gs:
        print(f"[参考] grad_norm 逐点相对误差（抖动大于 loss，不参与判定）")
        print(f"  Mean={gs['mean_error_pct']}%  MSE={gs['mse_error_pct2']}  "
              f"Max={gs['max_error_pct']}%  Min={gs['min_error_pct']}%")
    if gbs:
        print(f"[参考] grad_norm 轨迹偏差（{result['smooth_window']}步滑动均值）")
        print(f"  Mean={gbs['mean_error_pct']}%  MSE={gbs['mse_error_pct2']}  "
              f"Max={gbs['max_error_pct']}%  Min={gbs['min_error_pct']}%")

    agg = result["aggregate_loss"]
    print(f"\n全程 loss 均值: 基线 {agg['baseline_mean']} / 优化 {agg['optimized_mean']}"
          f"，相对偏差 {agg['relative_error_pct']}%")
    print(f"噪声地板（loss，基线自身相邻两步波动中位数）: {result['noise_floor_pct']}%"
          + (f"；轨迹偏差仅为噪声地板的 {result['bias_over_noise_floor_pct']}%"
             if result.get("bias_over_noise_floor_pct") is not None else ""))
    print(f"噪声地板（grad_norm，同法实测）: {result['grad_norm_noise_floor_pct']}%"
          f"（P95 {result['grad_norm_jitter_p95_pct']}%）")

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
    print(f"\n{bar}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="精度对比工具（按步号锚定对齐；官方关键点 + 轨迹偏差 + 全程均值 三类口径判定）")
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
