#!/bin/bash
# 抓训练失败的真错误
# 用法: bash whats_wrong.sh <日志文件路径>
# 为什么不能直接 tail：torchrun 失败时会打印约 25 行统一格式报告，
# 会把真正的 Python traceback 顶出 tail 的窗口
set -u

LOG="${1:-}"
if [ -z "$LOG" ]; then
  echo "用法: bash whats_wrong.sh <日志文件路径>"
  exit 1
fi
if [ ! -f "$LOG" ]; then
  echo "[ERROR] 找不到日志: $LOG"
  echo "  可用的日志:"
  ls -t /workspace/ascend_ws/MindSpeed-MM/logs/*.log 2>/dev/null | head -5 | sed 's/^/    /'
  exit 1
fi

echo "==================================================="
echo " 错误排查: $(basename "$LOG")"
echo " 日志行数: $(wc -l < "$LOG")"
echo "==================================================="

echo
echo "=== 1) 步数 ==="
echo "  iteration 行数: $(grep -c 'elapsed time per iteration' "$LOG")"

echo
echo "=== 2) 第一个 Python traceback（最多 50 行）==="
awk '/Traceback \(most recent call last\)/{f=1} f{print; n++} n>50{exit}' "$LOG"
echo "  （以上为空 = 没有 Python traceback，可能是别的问题）"

echo
echo "=== 3) 异常行 ==="
grep -n -iE "error:|RuntimeError|ImportError|ModuleNotFoundError|ValueError|TypeError|KeyError|FileNotFoundError|ChildFailedError|Permission denied" "$LOG" | head -20

echo
echo "=== 4) 日志最后 15 行 ==="
tail -15 "$LOG"

echo
echo "=== 5) 常见错误速查 ==="
hit=0
if grep -q "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" "$LOG"; then
  hit=1
  echo "  ⚠ 命中: 用了 triton 路径，但 triton-ascend 3.2.0 的 npu_utils.cpp 引用了"
  echo "           CANN 9.0.0 中不存在的 RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE"
  echo "     解决: 在仓库根执行 python3 ascend_porting/patch_triton.py（幂等）"
fi
if grep -q "skip_flash_attn_recompute cannot be True" "$LOG"; then
  hit=1
  echo "  ⚠ 命中: eager 与 skip_*_recompute 互斥"
  echo "     解决: 用 run_round.sh 重新生成配置，它会自动配好这两个开关"
fi
if grep -q "No module named 'megatron'" "$LOG"; then
  hit=1
  echo "  ⚠ 命中: 缺 NON_MEGATRON=true"
  echo "     解决: 命令前加 export NON_MEGATRON=true；用 run_round.sh 则已自带"
fi
if grep -q "No module named 'fla_npu'" "$LOG"; then
  hit=1
  echo "  ⚠ 命中: 用了 ascendc 但 fla_npu 没装"
  echo "     解决: 先跑 console/install_fla_npu.sh"
fi
if grep -q "Address already in use" "$LOG"; then
  hit=1
  echo "  ⚠ 命中: 6000 端口被占，说明已经有训练在跑"
  echo "     解决: ps -eo args | grep \"[m]indspeed_mm\" 看清后，不要重复启动"
fi
if grep -qE "NPU out of memory|OutOfMemory|out of memory" "$LOG"; then
  hit=1
  echo "  ⚠ 命中: 显存不足，可能有别的进程在占卡"
  echo "     解决: npu-smi info 看占用，确认没有第二个训练"
fi
if grep -q "No such file or directory" "$LOG"; then
  hit=1
  echo "  ⚠ 命中: 找不到文件（权重 / 数据集 / 软链接可能丢了）"
  echo "     解决: 跑 health_check.sh 看哪一项缺失，然后告诉队长"
fi
[ "$hit" = "0" ] && echo "  未命中已知错误模式"

echo
echo "→ 如果上面没能定位，把【完整输出】截图发给队长，"
echo "  并附上: 你执行的命令 + 这份完整输出 + health_check.sh 的输出"
