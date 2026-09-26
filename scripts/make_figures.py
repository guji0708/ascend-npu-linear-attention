#!/usr/bin/env python3
"""生成技术插图（本机运行，无需网络）。

产物:
  arch.png   系统技术架构图（含应用场景链路）
  loss.png   训练损失收敛曲线（A/C 两轮 1000 步，数据取自原始训练日志）

训练日志目录默认取仓库根的 logs/，可用环境变量 ASCEND_LOG_DIR 覆盖。

配色沿用统一视觉规范: #0F172A 主色 / #2563EB 蓝 / #059669 绿。
"""
import os
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for _cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
    if _cand in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
OUT = HERE / "out"
LOG_DIR = Path(os.environ.get("ASCEND_LOG_DIR", HERE.parent / "logs"))

INK, BLUE, GREEN = "#0F172A", "#2563EB", "#059669"
RE_ITER = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
RE_LOSS = re.compile(r"(?<![_\w])loss\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?)")


# ---------------------------------------------------------------- 图 1 架构图
def _txt_w(s):
    """粗估文本宽度（pt @ fontsize=1 的倍数），用于按内容分配列宽，避免文字互压。
    中文/全角按 1.0 计，ASCII 按 0.55 计 —— 只需相对比例正确。"""
    return sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in s)


def fig_arch(out_png):
    # 每层等高的「能力层」，算子优化层下方单独挂一条全宽指标条（strip）。
    layers = [
        {"name": "应用场景", "chips": ["大田种植实时巡检", "设施农业精准防控", "农技推广智能辅助"],
         "face": "#EFF6FF", "edge": BLUE},
        {"name": "决策闭环", "chips": ["① 拍照", "② 诊断", "③ 处方", "④ 多轮问答"],
         "face": "#ECFDF5", "edge": GREEN},
        {"name": "模型层", "chips": ["Qwen3.5-0.8B 多模态　FP16　32K 上下文"],
         "face": "#F1F5F9", "edge": INK},
        {"name": "算子优化层", "chips": ["GDN → AscendC", "Conv1d → Triton 融合", "跳重计算"],
         "face": "#FEF3C7", "edge": "#B45309",
         "strip": "实测 12498.1 → 3624.2 ms/步，加速比 3.45×（101–200 步口径）"},
        {"name": "训练框架", "chips": ["MindSpeed-MM（FSDP2）　单卡　GBS=8"],
         "face": "#F1F5F9", "edge": INK},
        {"name": "算力底座", "chips": ["Ascend910_9382", "CANN 9.0.0", "torch_npu 2.10.0"],
         "face": "#EFF6FF", "edge": BLUE},
    ]
    H, GAP, SH = 1.18, 0.30, 0.62
    rows_h = sum(H + (SH + 0.12 if L.get("strip") else 0) for L in layers) + GAP * (len(layers) - 1)
    fig, ax = plt.subplots(figsize=(10, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, rows_h + 1.5)
    ax.axis("off")

    x0, x1 = 1.70, 9.90
    cx0, cx1 = x0 + 2.45, x1 - 0.25  # 芯片区
    ax.text(5.8, rows_h + 1.40, "系统技术架构", ha="center", va="top",
            fontsize=15, fontweight="bold", color=INK)

    y = rows_h + 0.90
    mids, boxes = [], []
    for L in layers:
        ax.add_patch(FancyBboxPatch((x0, y - H), x1 - x0, H,
                                    boxstyle="round,pad=0,rounding_size=0.10",
                                    linewidth=1.4, edgecolor=L["edge"], facecolor=L["face"]))
        ax.text(x0 + 0.22, y - H / 2, L["name"], ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=L["edge"])
        chips = L["chips"]
        # 按各 chip 文本宽度比例分配列位置，长文本不会压到下一列
        wts = np.array([_txt_w(c) for c in chips], dtype=float)
        avail = cx1 - cx0
        pos, acc = [], 0.0
        for w in wts:
            pos.append(cx0 + acc)
            acc += avail * w / wts.sum()
        for px, c in zip(pos, chips):
            ax.text(px, y - H / 2, c, ha="left", va="center", fontsize=9.8, color="#1E293B")
        boxes.append((y, y - H))
        mids.append(y - H / 2)
        y -= H + GAP

        if L.get("strip"):
            # 指标条只占内容区宽度，左侧标签列留空给层间箭头，箭头不会穿过色块
            sy = y + GAP - 0.12
            ax.add_patch(FancyBboxPatch((cx0 - 0.10, sy - SH), x1 - cx0 + 0.10, SH,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        linewidth=0, facecolor=GREEN))
            ax.text((cx0 - 0.10 + x1) / 2, sy - SH / 2, L["strip"],
                    ha="center", va="center", fontsize=10.2, fontweight="bold", color="white")
            y -= SH + 0.12

    # 层间箭头只画在左侧标签列，不会穿过中间的指标条
    for (_, bot), (top, _) in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 1.00, bot - 0.05), (x0 + 1.00, top + 0.05),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.2, color="#94A3B8"))

    ax.text(5.8, 0.10, "全链路国产化：昇腾 NPU + CANN + MindSpeed-MM + 国产开源模型",
            ha="center", va="bottom", fontsize=9.5, color="#64748B")

    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  出图: {out_png}")


# ---------------------------------------------------------------- 图 2 收敛曲线
def parse_loss(path):
    steps, losses = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "iteration" not in line:
                continue
            mi, ml = RE_ITER.search(line), RE_LOSS.search(line)
            if not (mi and ml):
                continue
            steps.append(int(mi.group(1)))
            losses.append(float(ml.group(1)))
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(losses)[order]


def fig_loss(out_png):
    # 两条曲线几乎完全重合（这正是精度对齐的结论），故基线画粗、优化画细并叠在其上，
    # 使"红线被蓝线覆盖"这一视觉本身成为证据，同时两条都可见。
    series = [("Baseline (eager)", "train_eager1000.log", "#C0392B", 2.6, 0.85),
              ("AscendC", "train_ascendc1000.log", BLUE, 1.1, 1.0)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for label, fname, color, lw, alpha in series:
        p = LOG_DIR / fname
        if not p.exists():
            print(f"  [WARN] 缺日志 {p}")
            continue
        s, l = parse_loss(p)
        ax.plot(s, l, color=color, linewidth=lw, alpha=alpha,
                label=f"{label} · {len(s)} steps")
    ax.set_title("训练损失收敛曲线（A 基线 vs C 优化，1000 步）", fontsize=13,
                 fontweight="bold", color=INK)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1000)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  出图: {out_png}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_arch(OUT / "arch.png")
    fig_loss(OUT / "loss.png")


if __name__ == "__main__":
    main()
