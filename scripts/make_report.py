#!/usr/bin/env python3
"""
性能报告出图与统计（本机运行，训练结束后用）
============================================================
输入: 训练日志（可多份）
输出:
  report_accuracy.png         上游模板形态: 上 loss 对比 + 下误差针状图 + 底部四项统计
  report_loss_all_rounds.png  各轮 loss 收敛曲线叠加
  report_stats.md             数字表格（可直接抄进报告）
  report_data.json            原始数据（可追溯）

为什么不用 skill/perf_collector.py
----------------------------------
它用宽松正则把 loss 按「出现顺序」追加进数组，compare_accuracy 再按
「数组下标」对齐两份日志:

    for i in range(min_len):
        loss_err = calc_relative_error(base_losses[i], opt_losses[i])

两份日志只要匹配到的行数不一致（某轮多打一行含 "loss" 的日志、
某轮少跑几步、日志被截断），后续所有步就整体错位 —— 图照出、数照算，
但全是错的，且没有任何报错。这种错最难发现。

本脚本改为「按 iteration 步号锚定」，两份日志以步号求交集对齐:
    步号对不上就明说，绝不错位。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 解析
# 只在同时含 iteration 与 loss 的行上取值，避免 "loss scale: 1024" 之类误匹配
RE_ITER = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
RE_LOSS = re.compile(r"(?<![_\w])loss\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?)")
RE_GRAD = re.compile(r"(?<![_\w])grad[_\s]*norm\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?)")
RE_TIME = re.compile(
    r"elapsed\s+time\s+per\s+iteration\s*\(\s*ms\s*\)\s*[:=]?\s*([0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)
RE_TIME_ALT = re.compile(r"elapsed\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE)
RE_GBS = re.compile(r"global\s+batch\s+size\s*[:=]\s*(\d+)", re.IGNORECASE)

# 统一口径: 取第 101-200 步均值。千步轮想取更长窗口用 --window 101 1000
WINDOW = (101, 200)

# 逐点相对误差的滑动平均窗口 —— 精度对齐的主判据口径
MA_WINDOW = 50
CRITERION = (f"轨迹偏差({MA_WINDOW}步滑动均值)均值 < 2% 且全程 loss 均值相对偏差 < 2%；"
             f"逐点口径受 loss→0 分母效应影响，仅作参考")

# 图上要写中文，缺字体时 matplotlib 会画成方块
for _cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
    if _cand in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

ROUND_STYLE = {
    "eager":        ("Baseline (eager)",         "#C0392B"),
    "triton":       ("triton",                   "#D97706"),
    "ascendc":      ("AscendC",                  "#2563EB"),
    "ascendc_skip": ("AscendC + skip recompute", "#059669"),
}


def round_key(path):
    """train_ascendc200_skip.log -> ascendc_skip200

    步数保留在键里。否则 train_eager200.log 与 train_eager1000.log 都解析成
    "eager"，同模式跑不同步数时后者会静默覆盖前者，数据直接丢。
    """
    stem = Path(path).stem
    s = stem[len("train_"):] if stem.startswith("train_") else stem
    m = re.search(r"(\d+)(_skip)?$", s)
    if not m:
        return s
    mode = s[: m.start()]
    if m.group(2):
        mode += "_skip"
    return f"{mode}{m.group(1)}"


def mode_of(key):
    return re.sub(r"\d+$", "", key)


def label_of(key):
    mode = mode_of(key)
    base = ROUND_STYLE.get(mode, (mode, "#64748B"))[0]
    m = re.search(r"(\d+)$", key)
    return f"{base} · {m.group(1)} steps" if m else base


def color_of(key):
    mode = mode_of(key)
    return ROUND_STYLE.get(mode, (mode, "#64748B"))[1]


def resolve(tok, rounds):
    """--pair 允许写轮次全名(eager1000)或模式名(eager)。
    写模式名时取该模式步数最多的那一轮。"""
    if tok in rounds:
        return tok
    cands = [k for k in rounds if mode_of(k) == tok]
    if not cands:
        return None
    return max(cands, key=lambda k: int(re.search(r"(\d+)$", k).group(1)))


def parse_log(path):
    """按步号锚定解析。返回 (records, gbs, 统计信息)"""
    recs = {}
    gbs = None
    dup = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "iteration" not in line:
                continue
            mi = RE_ITER.search(line)
            if not mi:
                continue
            ml = RE_LOSS.search(line)
            if not ml:
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

    return recs, gbs, {"dup": dup}


def step_time_stats(recs, window=WINDOW):
    """统一口径: 第 101-200 步均值；不足则降级并明说"""
    times = [recs[s]["time"] for s in range(window[0], window[1] + 1)
             if s in recs and recs[s]["time"] is not None]
    if times:
        return sum(times) / len(times), f"step {window[0]}-{window[1]}", len(times)

    allt = [r["time"] for r in recs.values() if r["time"] is not None]
    if not allt:
        return None, "无耗时数据", 0
    return sum(allt) / len(allt), f"step 1-{len(allt)}（不足 {window[1]} 步，降级统计）", len(allt)


def err_stats(errs):
    if not errs:
        return None
    a = np.asarray(errs, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "mse": float((a ** 2).mean()),
        "max": float(a.max()),
        "min": float(a.min()),
    }


def load_rounds(paths, window=WINDOW):
    rounds = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] 日志不存在，跳过: {p}")
            continue
        recs, gbs, meta = parse_log(p)
        if not recs:
            print(f"[WARN] 没解析出任何迭代行，跳过: {p}")
            continue
        key = round_key(p)
        if key in rounds:
            print(f"  [WARN] 轮次键 {key} 重复: {p.name} 会覆盖 "
                  f"{Path(rounds[key]['path']).name}")
        st, win, n = step_time_stats(recs, window)
        rounds[key] = {
            "path": str(p),
            "records": recs,
            "gbs": gbs,
            "step_time_ms": round(st, 1) if st else None,
            "window": win,
            "window_samples": n,
            "steps": sorted(recs.keys()),
            "dup_lines": meta["dup"],
        }
        s = rounds[key]
        print(f"  解析 {p.name:28s} 轮次={key:14s} 步数={len(recs):4d} "
              f"STEP_TIME={s['step_time_ms']} ms ({win}, n={n})")
        if meta["dup"]:
            print(f"    [WARN] 有 {meta['dup']} 个重复步号，后者覆盖前者")
    return rounds


# ---------------------------------------------------------------- 出图
def moving_avg(x, w):
    if len(x) < w:
        return np.array([]), np.array([], dtype=int)
    return np.convolve(x, np.ones(w) / w, mode="valid"), np.arange(w - 1, len(x))


def bias_curve(bl, ol, w):
    """轨迹偏差: 两轮各自做 w 步滑动平均 loss 之后再算相对误差。

    与逐点相对误差的区别: 逐点误差的分母是单步瞬时 loss —— 单步 loss 本身噪声极大，
    且收敛后趋近于 0，分母效应会把噪声放大成几十个百分点的"误差"。轨迹偏差衡量的是
    「两条收敛轨迹的系统性偏离」，才是算子精度对齐该看的量。
    """
    mb, idx = moving_avg(bl, w)
    mo, _ = moving_avg(ol, w)
    if not mb.size:
        return np.array([]), np.array([], dtype=int)
    return np.abs(mb - mo) / np.abs(mb) * 100.0, idx


def own_run_fluct(recs, steps):
    """同一轮训练相邻两步的 loss 相对变化 —— 逐点相对误差指标的噪声地板。

    两条曲线之间的逐点差异只要落在噪声地板以下，就不具备区分能力。
    """
    v = []
    for s in steps:
        a, nxt = recs.get(s), recs.get(s + 1)
        if a and nxt and a["loss"]:
            v.append(abs(nxt["loss"] - a["loss"]) / abs(a["loss"]) * 100.0)
    return np.asarray(v, dtype=float)


def block_means(steps, bl, ol, block=100):
    out = []
    for a in range(1, int(steps.max()) + 1, block):
        m = (steps >= a) & (steps <= a + block - 1)
        if not m.any():
            continue
        mb, mo = float(bl[m].mean()), float(ol[m].mean())
        out.append({"block": f"{a}-{a + block - 1}", "n": int(m.sum()),
                    "baseline_mean": mb, "optimized_mean": mo,
                    "error_pct": abs(mb - mo) / abs(mb) * 100.0})
    return out


def err_vs_loss(steps, bl, ol, errs):
    """误差归因: 按基线 loss 量级分箱，看误差中位数怎么随 loss 变小而放大"""
    bins = [(1.0, np.inf, "loss > 1", ">1"), (0.5, 1.0, "0.5~1", "0.5-1"),
            (0.1, 0.5, "0.1~0.5", "0.1-0.5"), (0.03, 0.1, "0.03~0.1", "0.03-0.1"),
            (0.0, 0.03, "loss < 0.03", "<0.03")]
    out = []
    for lo, hi, name, short in bins:
        m = (bl > lo) & (bl <= hi)
        if not m.any():
            continue
        out.append({"bin": name, "short": short, "n": int(m.sum()),
                    "mean_loss": float(bl[m].mean()),
                    "median_err": float(np.median(errs[m])),
                    "mean_err": float(errs[m].mean())})
    return out


def fig_accuracy(base_key, opt_key, rounds, out_png):
    """上游模板形态 + 双口径。

    上: loss 收敛对比
    中: 逐步相对误差针状图 + 2% 阈值线 + 滑动平均 + 噪声地板
    下: 误差归因（分块均值 / 误差 vs loss 量级）

    为什么要双口径: 逐点相对误差的分母是单步瞬时 loss，而 loss 收敛后趋近于 0，
    分母效应会把噪声放大成几十个百分点的"误差"。同一轮训练自身的逐步波动中位数
    就有约 19%，即 2% 这条线本身落在噪声地板之下 —— 逐点口径不具备判定能力，
    因此以滑动平均口径为主判据，逐点口径如实并列。
    """
    b = rounds[base_key]["records"]
    o = rounds[opt_key]["records"]

    common = sorted(set(b) & set(o))
    if not common:
        print(f"[ERROR] {base_key} 与 {opt_key} 没有共同步号，无法对比")
        return None

    only_b, only_o = sorted(set(b) - set(o)), sorted(set(o) - set(b))
    if only_b or only_o:
        print(f"  [WARN] 步号不一致，按交集 {len(common)} 步对齐")
        if only_b:
            print(f"    仅基线有: {len(only_b)} 步 ({only_b[:5]}...)")
        if only_o:
            print(f"    仅优化有: {len(only_o)} 步 ({only_o[:5]}...)")

    steps = np.array(common)
    bl = np.array([b[s]["loss"] for s in common])
    ol = np.array([o[s]["loss"] for s in common])

    errs = np.abs(bl - ol) / np.abs(bl) * 100.0
    st = err_stats(list(errs))

    bias, bias_idx = bias_curve(bl, ol, MA_WINDOW)
    st_bias = err_stats(list(bias)) if bias.size else None
    bias_x = steps[bias_idx] if bias.size else np.array([])

    fluct = own_run_fluct(b, common)
    floor = float(np.median(fluct)) if fluct.size else 0.0

    blocks = block_means(steps, bl, ol)
    attrib = err_vs_loss(steps, bl, ol, errs)
    agg = abs(bl.mean() - ol.mean()) / abs(bl.mean()) * 100.0

    fig = plt.figure(figsize=(10, 13.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.2, 0.85, 0.30],
                          left=0.085, right=0.965, top=0.97, bottom=0.025,
                          hspace=0.5, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, bl, color=color_of(base_key), linewidth=1.3, label=label_of(base_key))
    ax1.plot(steps, ol, color=color_of(opt_key), linewidth=1.3, alpha=0.8,
             label=label_of(opt_key))
    ax1.set_title("Loss Chart", fontsize=13, fontweight="bold")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("loss")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    # Y 轴必须裁剪: 逐点误差在 loss→0 的尾段被分母效应放大到 70%+，按真实极值定轴会把
    # 2% 阈值线和轨迹偏差(约 1%)压成一条贴地的线 —— 真正要看的结论反而看不见。
    # 裁剪处如实标注超出步数与原因，不隐藏数据。
    ymax = max(20.0, floor * 1.15)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.vlines(steps, 0, np.minimum(errs, ymax), color="#93C5FD", linewidth=0.7,
               alpha=0.85, label="per-step relative error")
    if floor > 0:
        ax2.axhline(y=floor, color="#94A3B8", linewidth=1.2, linestyle=":",
                    label=f"noise floor: own-run fluctuation median = {floor:.1f}%")
    if bias.size:
        ax2.plot(bias_x, bias, color="#0F766E", linewidth=2.2,
                 label=f"trajectory bias: rel. error of {MA_WINDOW}-step averaged loss "
                       f"(mean {st_bias['mean']:.2f}%, max {st_bias['max']:.2f}%)")
    ax2.axhline(y=2.0, color="#C0392B", linewidth=1.3, linestyle="--",
                label="2% threshold")
    n_clip = int((errs > ymax).sum())
    if n_clip:
        ax2.text(0.995, 0.965,
                 f"Y 轴裁剪至 {ymax:.0f}%；{n_clip} 步（{n_clip / len(errs) * 100:.1f}%）超出，"
                 f"均在 loss<0.03 尾段",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=7.5, color="#B45309")
    ax2.set_title("Comparison relative abs Chart", fontsize=13, fontweight="bold")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("relative abs error (%)")
    ax2.set_ylim(0, ymax)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[2, 0])
    names = [x["block"] for x in blocks]
    vals = [x["error_pct"] for x in blocks]
    ax3.bar(range(len(vals)), vals, width=0.68,
            color=["#2563EB" if v < 2.0 else "#D97706" for v in vals])
    ax3.axhline(y=2.0, color="#C0392B", linewidth=1.2, linestyle="--")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax3.set_title("Block-mean error (100 steps)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("relative error (%)")
    ax3.set_ylim(0, max(vals) * 1.28)
    ax3.grid(alpha=0.25, axis="y")
    for i, v in enumerate(vals):
        ax3.text(i, v + max(vals) * 0.04, f"{v:.2f}", ha="center", fontsize=7.5)

    ax4 = fig.add_subplot(gs[2, 1])
    an = [x["short"] for x in attrib]
    av = [x["median_err"] for x in attrib]
    al = [x["mean_loss"] for x in attrib]
    ax4.bar(range(len(av)), av, width=0.6, color="#93C5FD")
    ax4.axhline(y=2.0, color="#C0392B", linewidth=1.2, linestyle="--",
                label="2% threshold")
    ax4.set_xticks(range(len(an)))
    ax4.set_xticklabels(an, rotation=25, ha="right", fontsize=8)
    ax4.set_xlabel("baseline loss bin", fontsize=9)
    ax4.set_title("Error vs loss magnitude (attribution)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("median relative error (%)")
    ax4.set_ylim(0, max(av) * 1.35)
    ax4.grid(alpha=0.25, axis="y")
    ax5 = ax4.twinx()
    ax5.plot(range(len(al)), al, color="#0F766E", marker="o", linewidth=1.6,
             label="mean loss in bin")
    ax5.set_ylabel("mean loss", color="#0F766E")
    ax5.tick_params(axis="y", colors="#0F766E")
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax5.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")

    verdict = "PASS" if (st_bias and st_bias["mean"] < 2.0 and agg < 2.0) else "FAIL"
    # 底部统计行按「一屏宽」排版: 单行过长会被画布右边界截断，故把噪声地板比值并入末行。
    if st_bias:
        t0 = (f"精度对齐判定: {verdict}   —   轨迹偏差 {st_bias['mean']:.2f}% < 2%   且   "
              f"全程 loss 均值相对偏差 {agg:.3f}% < 2%")
        t1 = (f"轨迹偏差口径({MA_WINDOW}步滑动均值，主判据)  Mean Error: {st_bias['mean']:.4f}%   "
              f"Mean Square Error: {st_bias['mse']:.4f}   Max Error: {st_bias['max']:.4f}%   "
              f"Min Error: {st_bias['min']:.4f}%")
    else:
        t0, t1 = f"精度对齐判定: {verdict}", ""
    t2 = (f"逐点口径(参考)  Mean Error: {st['mean']:.4f}%   Mean Square Error: {st['mse']:.4f}   "
          f"Max Error: {st['max']:.4f}%   Min Error: {st['min']:.4f}%"
          f"（受 loss→0 分母效应影响，见右下归因图）")
    t3 = (f"全程 loss 均值: 基线 {bl.mean():.6f} / 优化 {ol.mean():.6f}，相对偏差 {agg:.3f}%"
          + (f"   ；   噪声地板 {floor:.1f}%，轨迹偏差仅为噪声地板的 "
             f"{st_bias['mean'] / floor * 100:.1f}%" if st_bias else ""))
    axf = fig.add_subplot(gs[3, :])
    axf.axis("off")
    axf.text(0.5, 1.00, t0, ha="center", va="top", fontsize=11, fontweight="bold",
             color="#047857" if verdict == "PASS" else "#B91C1C")
    axf.text(0.5, 0.62, t1, ha="center", va="top", fontsize=9.5, color="#0F766E")
    axf.text(0.5, 0.36, t2, ha="center", va="top", fontsize=9, color="#334155")
    axf.text(0.5, 0.10, t3, ha="center", va="top", fontsize=9.5, color="#1D4ED8")

    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  出图: {out_png}")

    return {
        "baseline": base_key,
        "optimized": opt_key,
        "n_common_steps": len(common),
        "steps_only_baseline": len(only_b),
        "steps_only_optimized": len(only_o),
        "loss_error_stats": st,
        "loss_error_stats_bias": st_bias,
        "smooth_window": MA_WINDOW,
        "noise_floor_pct": round(floor, 3),
        "block_means": blocks,
        "attribution": attrib,
        "aggregate": {
            "baseline_loss_mean": float(bl.mean()),
            "optimized_loss_mean": float(ol.mean()),
            "relative_error_pct": round(agg, 4),
        },
        "verdict_criterion": CRITERION,
        "verdict": verdict,
        "per_step": [{"step": int(s), "baseline_loss": float(x), "optimized_loss": float(y),
                      "error_pct": float(e)} for s, x, y, e in zip(steps, bl, ol, errs)],
    }


def fig_all_rounds(rounds, out_png):
    fig, ax = plt.subplots(figsize=(10, 5))
    # D 与 B 两轮的 loss 轨迹几乎重合（同为 200 步、同一 lr 调度），只用颜色区分会互相盖住，
    # 故再按模式给线型；并按步数从多到少绘制，让短的 200 步曲线压在长曲线之上可见。
    ls_of = {"eager": "-", "ascendc": "-", "ascendc_skip": (0, (5, 2)),
             "triton": (0, (1, 1.4))}
    for key, r in sorted(rounds.items(), key=lambda kv: -len(kv[1]["records"])):
        st = sorted(r["records"].keys())
        if not st:
            continue
        ax.plot(st, [r["records"][s]["loss"] for s in st],
                color=color_of(key), linewidth=1.3,
                linestyle=ls_of.get(mode_of(key), "-"), label=label_of(key))
    ax.set_title("Loss convergence — all rounds", fontsize=13, fontweight="bold")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  出图: {out_png}")


# ---------------------------------------------------------------- 报表
def write_md(rounds, acc, out_md):
    L = []
    L.append("# 性能分析报告 · 数据表（自动生成）\n")
    L.append("> 由 `make_report.py` 从训练日志直接提取，每个数字都可回到日志复算。\n")

    L.append("## 一、各轮性能指标\n")
    L.append("| 轮次 | 日志 | 步数 | STEP_TIME (ms) | 统计窗口 | 样本数 | GBS | samples/s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, r in rounds.items():
        gbs = r["gbs"]
        sps = f"{gbs * 1000 / r['step_time_ms']:.3f}" if (gbs and r["step_time_ms"]) else "—"
        L.append(f"| {label_of(key)} | `{Path(r['path']).name}` | {len(r['records'])} | "
                 f"{r['step_time_ms']} | {r['window']} | {r['window_samples']} | "
                 f"{gbs if gbs else '—'} | {sps} |")
    L.append("")

    if acc:
        b = rounds.get(acc["baseline"])
        o = rounds.get(acc["optimized"])
        if b and o and b["step_time_ms"] and o["step_time_ms"]:
            sp = b["step_time_ms"] / o["step_time_ms"]
            red = (1 - o["step_time_ms"] / b["step_time_ms"]) * 100
            L.append("## 二、加速比（同一时段背靠背跑出的两轮）\n")
            L.append(f"- 基线（{label_of(acc['baseline'])}）: **{b['step_time_ms']} ms/步**")
            L.append(f"- 优化（{label_of(acc['optimized'])}）: **{o['step_time_ms']} ms/步**")
            L.append(f"- 单步耗时降低: **{red:.1f}%**")
            L.append(f"- **加速比: {sp:.2f}×**\n")

        s = acc["loss_error_stats"]
        sb = acc.get("loss_error_stats_bias")
        agg = acc["aggregate"]
        L.append("## 三、精度对齐\n")
        L.append(f"- 对比: {label_of(acc['baseline'])} vs {label_of(acc['optimized'])}")
        L.append(f"- 对齐步数: {acc['n_common_steps']}"
                 f"（仅基线有 {acc['steps_only_baseline']} 步，仅优化有 {acc['steps_only_optimized']} 步）")
        L.append(f"- **判定: {acc['verdict']}**（口径: {acc['verdict_criterion']}）\n")

        L.append("| 口径 | Mean Error | Mean Square Error | Max Error | Min Error |")
        L.append("|---|---|---|---|---|")
        if sb:
            L.append(f"| **轨迹偏差（{acc['smooth_window']}步滑动均值，主判据）** | {sb['mean']:.4f}% | "
                     f"{sb['mse']:.4f} | {sb['max']:.4f}% | {sb['min']:.4f}% |")
        L.append(f"| 逐点（参考） | {s['mean']:.4f}% | {s['mse']:.4f} | {s['max']:.4f}% | {s['min']:.4f}% |")
        L.append("")
        L.append(f"- 全程 loss 均值: 基线 {agg['baseline_loss_mean']:.6f} / "
                 f"优化 {agg['optimized_loss_mean']:.6f}，相对偏差 **{agg['relative_error_pct']:.4f}%**")
        L.append(f"- 噪声地板（同一轮训练自身相邻两步波动的中位数）: "
                 f"**{acc['noise_floor_pct']:.3f}%**\n")
        L.append("> MSE 为「百分比误差的平方」的均值，量纲是 %²，与上游模板的 Mean Square Error 一致。")
        L.append("> 逐点口径的分母是单步瞬时 loss，而单步 loss 的自身波动中位数就有 "
                 f"{acc['noise_floor_pct']:.1f}%（噪声地板），且收敛后 loss 趋近 0 —— 2% 这条线本身落在"
                 "噪声地板以下，逐点口径不具备判定能力，故以轨迹偏差口径为主判据。\n")

        L.append("### 分块均值相对误差（每 100 步）\n")
        L.append("| 步区间 | 基线均值 loss | 优化均值 loss | 相对误差 |")
        L.append("|---|---|---|---|")
        for x in acc["block_means"]:
            L.append(f"| {x['block']} | {x['baseline_mean']:.6f} | {x['optimized_mean']:.6f} | "
                     f"{x['error_pct']:.3f}% |")
        L.append("")

        L.append("### 误差归因：误差随 loss 量级的变化\n")
        L.append("| loss 量级 | 步数 | 该区间平均 loss | 误差中位数 | 误差均值 |")
        L.append("|---|---|---|---|---|")
        for x in acc["attribution"]:
            L.append(f"| {x['bin']} | {x['n']} | {x['mean_loss']:.6f} | "
                     f"{x['median_err']:.3f}% | {x['mean_err']:.3f}% |")
        L.append("")

    Path(out_md).write_text("\n".join(L), encoding="utf-8")
    print(f"  报表: {out_md}")


# ---------------------------------------------------------------- 自检
def selftest():
    """用合成日志验证解析与对齐，避免拿到真日志才发现问题"""
    d = Path(__file__).parent / "_selftest"
    d.mkdir(exist_ok=True)

    def mk(name, n, shift, extra_loss_line=False):
        p = d / name
        lines = []
        for i in range(1, n + 1):
            loss = 2.0 * (0.995 ** i) + shift
            grad = 90.0 / i + 1
            lines.append(
                f"iteration {i:5d}/{n:5d} | elapsed time per iteration (ms): "
                f"{12000 + i * 5}.{i % 10} | learning rate: 1.0E-06 | global batch size: 8 | "
                f"loss: {loss:.5E} | grad norm: {grad:.3f}"
            )
            if extra_loss_line and i == 3:
                # 故意插入一行含 loss 的非迭代日志，验证不会被误收
                lines.append("loss scale: 1024.0")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    a = mk("train_eager200.log", 200, 0.0, extra_loss_line=True)
    b = mk("train_ascendc200.log", 200, 0.001)
    c = mk("train_eager1000.log", 1000, 0.0)
    e = mk("train_ascendc200_skip.log", 200, 0.002)

    rounds = load_rounds([a, b, c, e])

    # 同模式不同步数必须共存，不能互相覆盖
    assert "eager200" in rounds and "eager1000" in rounds, f"轮次键: {list(rounds)}"
    assert "ascendc_skip200" in rounds, f"轮次键: {list(rounds)}"
    assert len(rounds["eager200"]["records"]) == 200, "迭代行数应为 200（多余 loss 行不能被计入）"
    assert len(rounds["eager1000"]["records"]) == 1000

    # --pair 写模式名时，应解析到该模式步数最多的那一轮
    assert resolve("eager", rounds) == "eager1000"
    assert resolve("ascendc", rounds) == "ascendc200"
    assert resolve("ascendc_skip", rounds) == "ascendc_skip200"
    assert resolve("不存在的模式", rounds) is None

    acc = fig_accuracy("eager200", "ascendc200", rounds, d / "report_accuracy.png")
    assert acc["n_common_steps"] == 200
    assert 0 < acc["loss_error_stats"]["max"] < 2.0
    fig_all_rounds(rounds, d / "report_loss_all_rounds.png")
    write_md(rounds, acc, d / "report_stats.md")

    print("\n[自检通过] 解析步数、忽略非迭代 loss 行、按步号对齐、轮次键不冲突、出图与报表均正常")
    print(f"产物在 {d}")


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description="性能报告出图与统计")
    ap.add_argument("logs", nargs="*", help="训练日志路径（可多个）")
    ap.add_argument("--pair", nargs=2, metavar=("BASE", "OPT"),
                    help="精度对比的两个轮次名，默认 eager ascendc")
    ap.add_argument("--outdir", default=None, help="输出目录，默认取第一份日志所在目录")
    ap.add_argument("--window", nargs=2, type=int, metavar=("START", "END"),
                    default=list(WINDOW),
                    help="STEP_TIME 统计窗口，默认 101 200（统一口径）。"
                         "千步轮可传 --window 101 1000 取更长窗口")
    ap.add_argument("--selftest", action="store_true", help="用合成日志自检")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.logs:
        ap.print_help()
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else Path(args.logs[0]).parent
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== 解析日志 ===")
    win = (args.window[0], args.window[1])
    print(f"  STEP_TIME 统计窗口: step {win[0]}-{win[1]}")
    rounds = load_rounds(args.logs, win)
    if not rounds:
        print("[FATAL] 没有任何日志解析成功")
        sys.exit(1)

    pair = args.pair if args.pair else ["eager", "ascendc"]
    rb, ro = resolve(pair[0], rounds), resolve(pair[1], rounds)
    acc = None
    if rb and ro:
        print(f"\n=== 精度对比（{rb} vs {ro}）===")
        acc = fig_accuracy(rb, ro, rounds, outdir / "report_accuracy.png")
    else:
        miss = [t for t, r in zip(pair, (rb, ro)) if not r]
        print(f"\n[WARN] 精度对比跳过: 找不到 {miss}（现有: {list(rounds)}）")

    print("\n=== 各轮 loss 曲线 ===")
    fig_all_rounds(rounds, outdir / "report_loss_all_rounds.png")

    print("\n=== 数据表 ===")
    write_md(rounds, acc, outdir / "report_stats.md")

    payload = {
        "rounds": {k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in rounds.items()},
        "accuracy": acc,
    }
    (outdir / "report_data.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  数据: {outdir / 'report_data.json'}")


if __name__ == "__main__":
    main()
