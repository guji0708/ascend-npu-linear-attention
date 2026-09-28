# 昇腾 NPU 线性注意力算子优化与多模态微调

> 把 Qwen3.5-0.8B 多模态模型迁移到昇腾 NPU，并把计算最重的线性注意力算子由纯 PyTorch 路径换成昇腾原生 AscendC 实现。

[English](#english) | 中文

---

## 这是什么

一套**可运行、可复现**的昇腾 NPU 微调流水线，以及为跑通它所沉淀的**环境适配工具链**。

核心命题：让大模型在国产算力上跑得动、跑得稳、跑得快。为此本项目完成了两件事：

1. **把 Qwen3.5-0.8B 多模态模型完整迁移到昇腾 NPU**（MindSpeed-MM + FSDP2）；
2. **对计算最重的线性注意力算子做算子级替换**：GDN（Gated DeltaNet）由纯 PyTorch 的 `eager` 路径切换到昇腾原生 **AscendC** 实现。

`skill/` 里带一条完整的示例链路，用来证明整条路能跑通：以**作物病虫害图像问答**为场景，从数据构造、领域微调、指标采集一直到 Gradio 图文问答演示，全程可执行。示例场景与算子优化是两件独立的事——算子层可以脱离示例单独复现。

## 实测结果

**性能**（统一口径：第 101–200 步单步耗时均值）

| 轮次 | 算子配置 | `STEP_TIME` | `samples/s` | 加速比 |
|---|---|---|---|---|
| A 基线 | GDN=`eager` / Conv1d=`eager` | 12498.1 ms | 0.640 | 1.00× |
| **C 优化** | GDN=`ascendc` / Conv1d=`triton_with_transpose` | **3624.2 ms** | **2.207** | **3.45×** |
| D 对照 | 同 C + 跳过重计算 | 3783.9 ms | 2.114 | 3.30× |
| B 参考 | GDN=`triton` / Conv1d=`triton` | 4071.7 ms | 1.965 | 3.07× |

单步耗时降低 **71.0%**；长窗口复核（第 101–1000 步，900 样本）为 12165.4 → 3513.5 ms，加速比 **3.46×**，结论不依赖窗口选择。

**精度**（A 基线 vs C 优化，1000 步逐步对齐）

| 口径 | Mean Error | MSE | Max Error | Min Error |
|---|---|---|---|---|
| 官方关键点（第 50 / 100 步） | **0.3239%** | 0.1733 | 0.5854% | 0.0623% |
| 轨迹偏差（50 步滑动均值，主判据） | **1.0613%** | 2.3604 | 4.6311% | 0.0001% |
| 逐点相对误差（参考） | 3.9181% | 55.9654 | 72.7794% | 0.0014% |

判定 **通过（三项判据全部达标）**：官方关键点第 50 步 0.0623% / 第 100 步 0.5854%，轨迹偏差均值 1.0613%，全程 loss 均值相对偏差 0.2205%，均低于 2% 阈值。

> 为什么以这三项为判据：逐点相对误差的分母是单步瞬时 loss，而单步 loss 自身波动中位数（噪声地板）就有 **16.279%**，且收敛后 loss 趋近 0。**2% 这条阈值线本身落在噪声地板以下**，逐点口径不具备判定能力；`grad_norm` 同频抖动更大（自身噪声地板 **16.248%**、P95 68.4%），同样只作参考。详见 `docs/环境适配问题与解法.md`。

## 环境基线

| 项目 | 版本 |
|---|---|
| NPU | Ascend910_9382（A3 档，64 GB HBM） |
| CANN | 9.0.0 |
| Python | 3.11.15 |
| PyTorch / torch_npu | 2.10.0 / 2.10.0 |
| 训练框架 | MindSpeed-MM（FSDP2 路线，`NON_MEGATRON=true`） |
| 算子库 | fla_npu（AscendC 算子），需从源码编译 |

## 快速开始

`fla_npu` 是 AscendC 算子库，必须先装好才能跑 `ascendc` 优化轮。该脚本处理了 CANN 9.0.0 下的 Stable-ABI 注册问题：

```bash
bash console/install_fla_npu.sh      # 从源码编译，25-40 分钟
```

若手上已有同架构环境导出的产物包（`fla_npu_artifacts.tar.gz`，打包与还原方法见 `docs/fla_npu产物清单与还原方法.md`），可直接还原，约 1 分钟：

```bash
bash console/restore_fla_npu.sh
```

再装 Python 依赖并跑主流程：

```bash
pip install -r requirements.txt

# 一键流程：模型下载 → 权重转换 → 配置生成 → 训练 → 指标采集
cd skill
python main.py --all --npus 1 \
    --data-dir /path/to/dataset/COCO2017 \
    --hf-dir   /path/to/ckpt/hf_path/Qwen3.5-0.8B \
    --dcp-dir  /path/to/ckpt/dcp_path/Qwen3.5-0.8B
```

也可分步执行：`--download` / `--convert` / `--config-gen` / `--train` / `--collect`。

> **从空环境到跑出可复算指标的完整步骤**（软件环境 → 资产 → 算子 → 自检 → 训练 → 复算）见 [`docs/复现指南.md`](docs/复现指南.md)。
> 仓库内脚本默认工作根目录为 `/workspace/ascend_ws`，可用环境变量 `ASCEND_WORK` 覆盖。

`main.py` 会自动探测 `fla_npu` 是否可用：探测到则走 `ascendc`，否则回落到 `eager`。
（要用 `triton` 需先打 `ascend_porting/patch_triton.py` 补丁 —— triton-ascend 3.2.0 的 `npu_utils.cpp` 引用了 CANN 9.0.0 中不存在的枚举，见下文问题 2。）

### 复现指标

```bash
# 统一口径性能指标（第 101-200 步均值 + 加速比）
bash scripts/summary_metric.sh logs/train_eager1000.log logs/train_ascendc1000.log

# 精度对齐：四项误差统计 + 对比图
python skill/compare_accuracy.py logs/train_eager1000.log logs/train_ascendc1000.log \
    --output accuracy_report.json
```

两个工具都**按 iteration 步号锚定对齐**，而不是按日志出现顺序追加——后者会把 `loss scale: 1024.0` 这类非迭代日志当成一步 loss，导致后续所有步静默错位。

## 目录结构

```
├── skill/                        可运行的微调 Skill（作物图像问答示例）
│   ├── main.py                   主流水线：下载 → 转换 → 配置 → 训练 → 采集
│   ├── prepare_data.py           数据集预处理（random.seed(42) 是两轮可比的前提）
│   ├── compare_accuracy.py       精度对齐（三类口径判定）
│   ├── perf_collector.py         指标提取（统一口径）
│   └── demo.py                   Gradio 图文问答演示
├── configs/                      训练配置
│   ├── qwen3_5_0.8B_config.yaml  0.8B 主实验配置
│   ├── qwen3_5_4B_config.yaml    4B 扩展配置
│   ├── finetune_qwen3_5_0.8B.sh  微调启动脚本
│   └── install_extensions.sh     扩展安装入口
├── ascend_porting/               昇腾环境适配工具链（本项目的工程沉淀）
│   ├── patch_fla_npu.py          fla_npu 源码补丁
│   ├── patch_triton.py           triton-ascend × CANN 9.0.0 兼容补丁
│   ├── patch_mindspeed_kwargs.py MindSpeed-MM 传参修正（AST 级定位）
│   ├── npu_ops_compat.py         算子回落与传参兼容垫片
│   ├── gawk_shim.sh              gawk 替身（镜像只带 mawk，构建脚本依赖 strftime）
│   └── common.sh                 公共变量与工具函数
├── console/                      环境安装与运维
│   ├── install_fla_npu.sh        fla_npu 源码编译与安装（Stable-ABI 路线）
│   ├── restore_fla_npu.sh        还原预编译算子产物（省 25-40 分钟编译）
│   ├── rebuild_env.sh            从零重建环境
│   ├── health_check.sh           环境自检
│   ├── watch_progress.sh         训练进度监控
│   └── whats_wrong.sh            故障诊断
├── scripts/                      消融驱动与指标复算
│   ├── run_round.sh              单轮训练驱动
│   ├── summary_metric.sh         统一口径指标复算（与上游示例脚本口径一致）
│   ├── make_report.py            出图与统计（按步号锚定对齐）
│   └── make_figures.py           架构图 / 收敛曲线
└── docs/
    ├── 复现指南.md               从空环境到跑出可复算指标的全过程 ← 先读这个
    ├── 环境适配问题与解法.md       5 个阻断性问题的因果链
    ├── fla_npu产物清单与还原方法.md 编译产物清单、打包与还原方法
    ├── 踩坑全记录.md              调试过程的完整记录
    └── figures/                  结果图（精度对齐、四轮收敛、技术架构）
```

## 结果图

- `docs/figures/report_accuracy.png` —— 精度对齐：Loss 收敛对比 + 逐步相对误差（含 2% 阈值线、噪声地板与官方关键点标注）+ 轨迹偏差 + 误差归因 + 三类口径四项统计
- `docs/figures/report_loss_all_rounds.png` —— 各轮 Loss 收敛曲线叠加
- `docs/figures/arch.png` —— 系统技术架构

三张图都由 `scripts/` 下的脚本从原始训练日志直接生成，可复算。

## 环境适配：5 个阻断性问题

把 `ascendc` 路径真正跑通，一共要解决 5 个卡死问题。完整因果链见 `docs/环境适配问题与解法.md`。

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | `gawk: command not found` | 镜像仅有 `mawk`，构建脚本用 `strftime` 输出时间戳 | `ascend_porting/gawk_shim.sh` |
| 2 | `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE is not a member of rtLimitType_t` | triton-ascend 3.2.0 引用了 CANN 9.0.0 尚未定义的 SIMT 枚举 | `ascend_porting/patch_triton.py` |
| 3 | `torch.ops.npu.npu_recompute_w_u_fwd` 不存在 | 算子需经 Stable-ABI 扩展注册，而该扩展从未构建 | 编译 `libfla_npu_stable.so` |
| 4 | legacy 扩展要求 `torch_npu>=2.10.0.post2`，实际为 `2.10.0` | 版本门槛不满足 | 改用 Stable-ABI 路线 + 运行期 ctypes 回落 |
| 5 | MindSpeed-MM 向 fla_npu 传参形参名不匹配 | 框架与算子库版本间接口漂移 | `ascend_porting/patch_mindspeed_kwargs.py` |

`docs/fla_npu产物清单与还原方法.md` 记录了编译产物的完整清单与还原方法，可在同架构新环境上省去 25–40 分钟的重编译。

## 消融设计

各轮**背靠背相邻执行**（中间不插入其他任务），共用同一张卡，避免跨时段环境竞争噪声。除下表字段外配置完全一致：

| 字段 | A 基线 | A' 隔离 | C 优化 | D 优化+跳重计算 | B 参考 |
|---|---|---|---|---|---|
| `gdn_implementation` | `eager` | `eager` | `ascendc` | `ascendc` | `triton` |
| `causal_conv1d_implementation` | `eager` | `triton` | 由框架升级为 `triton_with_transpose` | 同 C | `triton` |
| `skip_gdn_recompute` | `False` | `False` | `False` | `True` | `False` |
| `train_iters` | 1000 | 200 | 1000 | 200 | 200 |

**A→A' 隔离 Causal Conv1d 的贡献，A'→C 隔离 GDN 的贡献，C→D 隔离跳过重计算的贡献。**

A→C 一轮同时更换了 GDN 与 Causal Conv1d 两项实现，因此「3.45× 全部来自 GDN」只能算推断。A' 轮把算子替换的收益拆到**单个算子**：三轮同机背靠背重跑 200 步后，**GDN 的 AscendC 实现贡献 3.55×**，而 **Causal Conv1d 由 `eager` 换 `triton` 未观察到正收益（0.98×）**——它由 `triton` 升为 `triton_with_transpose` 是框架为匹配 AscendC GDN 的数据布局而强制的。故加速主体确认来自 GDN 的 AscendC 实现。

两点必须说明的限制：

- `causal_conv1d` 框架硬性规定 `gdn=ascendc` 时不得为 `eager`，且会自动升级为 `triton_with_transpose` 以匹配算子间数据布局；
- D/B 轮为 200 步，框架 lr 调度按总步数归一化（峰值在第 50 步），与 A/C（峰值在第 100 步）不同，**故 D/B 的 loss 不可与 A/C 直接比较**，只比 `STEP_TIME`（与 lr 无关）。

## 说明

- 本仓库代码面向昇腾 NPU 环境，`torch_npu` / CANN 为硬性前置。`compare_accuracy.py` 与 `perf_collector.py` 是纯日志/数值处理，CPU 环境也能跑。
- `prepare_data.py` 的 `random.seed(42)` **不可删除或修改**——它是两轮训练数据顺序一致、精度对比成立的前提。
- 训练期间请勿同时进行大文件传输或解压，实测会造成 14–18% 的耗时噪声，导致加速比失真。

---

## English

Porting **Qwen3.5-0.8B** (multimodal) to Ascend via MindSpeed-MM + FSDP2, and replacing the heaviest linear-attention operators with native AscendC implementations. The **GDN (Gated DeltaNet)** layer moves from the pure-PyTorch `eager` path to a fused **AscendC** operator.

`skill/` ships a complete end-to-end example (crop disease image QA) covering data preparation, domain fine-tuning, metric collection and a Gradio demo — evidence that the whole pipeline runs, not just the operator layer.

**Results** (common window: mean step time over iterations 101–200):

| Round | Operator config | `STEP_TIME` | Speedup |
|---|---|---|---|
| A baseline | GDN=`eager` | 12498.1 ms | 1.00× |
| **C optimized** | GDN=`ascendc` | **3624.2 ms** | **3.45×** |
| B reference | GDN=`triton` | 4071.7 ms | 3.07× |

**71.0%** lower step time. Accuracy alignment over 1000 aligned steps passes on all three criteria: official key steps (**0.0623%** at step 50, **0.5854%** at step 100), trajectory bias **1.0613%** (50-step moving average), and aggregate loss deviation **0.2205%** — all under the 2% threshold.

Also included: a porting toolchain (`ascend_porting/`) resolving five blocking CANN 9.0.0 issues, and a documented restore procedure for the compiled `fla_npu` operator artifacts.

```bash
pip install -r requirements.txt
cd skill && python main.py --all --npus 1 --data-dir <dataset> --hf-dir <hf_weights> --dcp-dir <dcp_weights>
```

`main.py` auto-detects `fla_npu`; if absent it falls back to `eager`, the zero-dependency baseline. `triton` is also usable, but only after applying `ascend_porting/patch_triton.py` (see issue #2 above).

---

## 第三方依赖与许可

本仓库的 Apache-2.0 仅覆盖独立开发的代码与文档。下列组件是外部依赖，不随本仓库分发，各自遵循其原始许可证：

| 组件 | 在本项目中的角色 | 许可证 |
|---|---|---|
| `flash-linear-attention-npu` | AscendC 线性注意力算子库（GDN 算子来源） | 原创代码 BSD-3-Clause |
| MindSpeed-MM | 昇腾多模态训练框架 | 昇腾官方页面声明为 Apache-2.0，以仓库内 LICENSE 原文为准 |
| triton-ascend | 昇腾 Triton 后端 | 继承上游 Triton 的 MIT |
| CANN / torch_npu | 昇腾基础软件栈 | 华为软件许可协议 |
| Qwen3.5-0.8B | 基座模型 | 见模型主页；本仓库不分发权重 |

`ascend_porting/` 下的补丁脚本在运行时读取并修改上述组件的源码（按字符串标记插入或删除），不复制、不再分发其源码，因此不构成衍生作品；补丁中新增的代码为原创。`console/install_fla_npu.sh` 从上游仓库克隆源码，不附带任何上游代码副本。

模型权重、数据集、编译产物（`*.so` / `*.whl` / `*.tar.gz`）均已列入 `.gitignore`，不入库。

## 许可

本项目以 **Apache License 2.0** 发布，全文见 [`LICENSE`](LICENSE)。选择 Apache-2.0 的原因：与昇腾生态（CANN、MindSpeed 等）主流许可证一致，含明确专利授权条款，且与上游 `flash-linear-attention-npu` 的多许可模式（原创代码 BSD-3-Clause）兼容，便于后续向社区回馈修改。
