#!/bin/bash
# 盯训练进度 —— 每分钟报一次，训练结束自动退出
# 用法: bash watch_progress.sh <日志文件路径>
set -u

LOG="${1:-}"
if [ -z "$LOG" ]; then
  echo "用法: bash watch_progress.sh <日志文件路径>"
  echo "例如: bash watch_progress.sh /workspace/c4ai/MindSpeed-MM/logs/train_eager_200.log"
  exit 1
fi
if [ ! -f "$LOG" ]; then
  echo "[ERROR] 找不到日志文件: $LOG"
  echo "  可能这条训练还没启动，或者路径写错了。可用的日志："
  ls -t /workspace/c4ai/MindSpeed-MM/logs/*.log 2>/dev/null | head -5 | sed 's/^/    /'
  exit 1
fi

TOTAL=200
echo "==================================================="
echo " 盯进度中（每分钟报一次，训练结束后自动退出）"
echo " 日志: $LOG"
echo "==================================================="

# [m] 那个方括号是技巧：不会匹配到本脚本自己的命令行
while ps -eo args | grep -q "[m]indspeed_mm/fsdp/train/trainer.py"; do
  n=$(grep -c 'elapsed time per iteration' "$LOG" 2>/dev/null || echo 0)
  if [ "$n" -gt 5 ]; then
    avg=$(grep 'elapsed time per iteration' "$LOG" \
      | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' \
      | awk -F '|' '{print$1}' | tail -n 50 \
      | awk '{s+=$1;c++} END {if(c>0) printf "%.1f", s/c}')
    awk -v a="$avg" -v n="$n" -v t="$(date +%H:%M:%S)" -v T="$TOTAL" \
      'BEGIN{r=(T-n)*a/1000; printf "[%s] %d/%d 步 | 近50步均 %s ms | 预计还需 %d 分\n", t, n, T, a, r/60}'
  else
    echo "[$(date +%H:%M:%S)] $n/$TOTAL 步（模型加载 / 预热中，正常）"
  fi
  sleep 60
done

echo
echo "==================================================="
echo " 训练进程已结束  $(date +%H:%M:%S)"
echo " 总步数: $(grep -c 'elapsed time per iteration' "$LOG" 2>/dev/null)"
echo "==================================================="
echo
echo "--- 最后 10 行 ---"
tail -10 "$LOG"
echo
echo "→ 出官方口径指标:"
echo "    bash scripts/summary_official.sh $LOG"
echo "→ 若最后几行是报错，用:"
echo "    bash console/whats_wrong.sh $LOG"
