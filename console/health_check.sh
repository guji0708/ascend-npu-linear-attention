#!/bin/bash
# 环境自检 —— 每次开机后先跑这个
# 用法: bash health_check.sh
set -u
REPO="${C4AI_WORK:-/workspace/c4ai}/MindSpeed-MM"
PY=/usr/local/python3.11.15/bin/python
export NON_MEGATRON=true
ok=1

echo "==================================================="
echo " 慧眼·智慧农业 —— 昇腾环境自检"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================="

echo
echo "=== 1) NPU 设备 ==="
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>/dev/null | sed -n '1,8p'
else
  echo "  [ERROR] 找不到 npu-smi"; ok=0
fi

echo
echo "=== 2) Python / torch / transformers ==="
$PY - <<'PYEOF' 2>&1 | tail -5
import torch, torch_npu, transformers
print("  python      :", __import__("sys").executable)
print("  torch       :", torch.__version__)
print("  torch_npu   :", torch_npu.__version__)
print("  transformers:", transformers.__version__)
print("  npu 可用    :", torch_npu.npu.is_available(), "| 设备:", torch_npu.npu.get_device_name(0))
PYEOF

echo
echo "=== 3) 关键文件 ==="
for f in "$REPO/ckpt/hf_path/Qwen3.5-0.8B/config.json" \
         "$REPO/ckpt/dcp_path/Qwen3.5-0.8B/latest_checkpointed_iteration.txt" \
         "$REPO/dataset/annotations_slim.json" \
         "$REPO/MindSpeed-MM" \
         "$REPO/mindspeed_mm/fsdp/train/trainer.py" \
         "${C4AI_WORK:-/workspace/c4ai}/skill/main.py"; do
  if [ -e "$f" ]; then
    echo "  OK    $f"
  else
    echo "  缺失  $f"
    ok=0
  fi
done

echo
echo "=== 4) import mindspeed_mm ==="
if cd "$REPO" 2>/dev/null; then
  $PY -c "import mindspeed_mm; print('  import OK')" 2>&1 | tail -4
else
  echo "  [ERROR] 找不到 $REPO"; ok=0
fi

echo
echo "=== 5) fla_npu（ascendc 优化轮需要）==="
$PY -c "import importlib.util; print('  fla_npu:', 'installed —— 可以跑 ascendc 优化轮' if importlib.util.find_spec('fla_npu') else 'NOT installed —— 优化轮前需先装')"

echo
echo "=== 6) 磁盘 ==="
df -h /workspace | tail -1 | sed 's/^/  /'

echo
echo "==================================================="
if [ "$ok" = "1" ]; then
  echo " 自检通过，可以开始干活"
else
  echo " 自检未通过：上面标了「缺失」或「ERROR」的项"
  echo " 请把整段输出截图发给队长，不要自己动手修"
fi
echo "==================================================="
