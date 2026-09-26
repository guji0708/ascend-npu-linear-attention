#!/bin/bash
# 跑一轮训练（200 步，统一性能口径需要 >= 200 步）
# 用法: bash run_round.sh <eager|ascendc>
#   eager   —— 基线轮（纯 PyTorch 实现，无需额外算子库）
#   ascendc —— 优化轮（需先装 fla_npu）
set -u

MODE="${1:-}"
if [ "$MODE" != "eager" ] && [ "$MODE" != "ascendc" ]; then
  echo "用法: bash run_round.sh <eager|ascendc>"
  echo "  eager   —— 基线轮（纯 PyTorch 实现）"
  echo "  ascendc —— 优化轮（需先装 fla_npu）"
  exit 1
fi

REPO="${ASCEND_WORK:-/workspace/ascend_ws}/MindSpeed-MM"
PY=/usr/local/python3.11.15/bin/python
ITERS=200
LOG="$REPO/logs/train_${MODE}_${ITERS}.log"
CFG="examples/qwen3_5/qwen3_5_0.8B_${MODE}_config.yaml"

cd "$REPO" || { echo "[ERROR] 找不到 $REPO"; exit 1; }
export NON_MEGATRON=true

# ---------------------------------------------------------------
# 安全闸门：任何一条不满足都不启动，避免浪费时间或搞坏环境
# ---------------------------------------------------------------
echo "=== 0) 安全闸门 ==="

if ps -eo args | grep -q "[m]indspeed_mm/fsdp/train/trainer.py"; then
  echo "  [ERROR] 已有训练在跑，不能重复启动（会撞 6000 端口 + 抢显存）"
  echo "  先等它结束，或联系队长"
  exit 1
fi
echo "  OK  没有其他训练在跑"

if [ "$MODE" = "ascendc" ]; then
  if ! $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('fla_npu') else 1)"; then
    echo "  [ERROR] 选了 ascendc 但 fla_npu 没装"
    echo "  先跑: bash console/install_fla_npu.sh"
    exit 1
  fi
  echo "  OK  fla_npu 已安装"
fi

for f in ckpt/dcp_path/Qwen3.5-0.8B/latest_checkpointed_iteration.txt \
         ckpt/hf_path/Qwen3.5-0.8B/config.json \
         dataset/annotations_slim.json; do
  if [ ! -e "$f" ]; then
    echo "  [ERROR] 缺文件: $f —— 停下，告诉队长"
    exit 1
  fi
done
echo "  OK  权重与数据集齐备"

# ---------------------------------------------------------------
echo
echo "=== 1) 生成配置（$MODE / train_iters=$ITERS）==="
$PY - "$MODE" "$ITERS" "$CFG" <<'PYEOF'
import sys, pathlib, yaml
mode, iters, out = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])

src = pathlib.Path("examples/qwen3_5/qwen3_5_4B_config.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

cfg["parallel"]["fully_shard_parallel_size"] = 1
cfg["model"]["model_name_or_path"] = "./ckpt/hf_path/Qwen3.5-0.8B"
cfg["model"]["gdn_implementation"] = mode
# ascendc 时 causal_conv1d 只支持 triton/ascendc；eager 时用 eager
cfg["model"]["causal_conv1d_implementation"] = "triton" if mode == "ascendc" else "eager"
# skip_*_recompute 与 eager 互斥，否则 overwrite_transformer_config 直接抛 ValueError
if mode != "ascendc":
    cfg["model"]["skip_gdn_recompute"] = False
    cfg["model"]["skip_flash_attn_recompute"] = False

dp = cfg["data"]["dataset_param"]
dp["preprocess_parameters"]["model_name_or_path"] = "./ckpt/hf_path/Qwen3.5-0.8B"
dp["basic_parameters"]["dataset_dir"] = "./dataset"
dp["basic_parameters"]["dataset"] = "./dataset/annotations_slim.json"
dp["basic_parameters"]["max_samples"] = 1000

cfg["training"]["gradient_accumulation_steps"] = 8
cfg["training"]["train_iters"] = iters
cfg["training"]["load"] = "./ckpt/dcp_path/Qwen3.5-0.8B"
# 两轮各存各的检查点，避免互相覆盖
cfg["training"]["save"] = "./save_path_" + mode

out.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True,
                        sort_keys=False), encoding="utf-8")
print("  已生成:", out)
PYEOF

echo
echo "=== 2) 关键字段核对 ==="
grep -n "gdn_implementation\|causal_conv1d_implementation\|skip_gdn_recompute\|skip_flash_attn_recompute\|train_iters\|fully_shard_parallel_size" "$CFG" | sed 's/^/  /'

echo
echo "=== 3) 启动训练（后台运行，可关浏览器）==="
mkdir -p logs
nohup bash -c "
  source /usr/local/Ascend/cann/set_env.sh
  export NON_MEGATRON=true
  export MULTI_STREAM_MEMORY_REUSE=2 TASK_QUEUE_ENABLE=2 ASCEND_LAUNCH_BLOCKING=0
  export ACLNN_CACHE_LIMIT=100000 CPU_AFFINITY_CONF=1
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  cd $REPO
  torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 \
    --master_addr localhost --master_port 6000 \
    mindspeed_mm/fsdp/train/trainer.py $CFG
" > "$LOG" 2>&1 &

sleep 5
if ps -eo args | grep -q "[m]indspeed_mm/fsdp/train/trainer.py"; then
  echo "  已启动 ✓"
else
  echo "  [WARN] 启动后没看到训练进程，稍等几秒再确认："
  echo "    ps -eo args | grep \"[m]indspeed_mm\""
fi

echo
echo "  日志文件: $LOG"
echo
echo "==================================================="
echo " 接下来复制这一行盯进度（每分钟报一次，结束自动退出）："
echo
echo "  bash console/watch_progress.sh $LOG"
echo
echo "==================================================="
