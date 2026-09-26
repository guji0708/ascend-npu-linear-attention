#!/bin/bash
# 出统一口径性能指标（可一次给多份日志，两份时自动算加速比）
# 口径与上游启动脚本一致：grep "elapsed time per iteration" | head -n 200 | tail -n 100
# 即取第 101-200 步平均值；吞吐量 = GBS × 1000 / STEP_TIME(ms)
# 用法: bash summary_metric.sh <日志1> [日志2 ...]
set -u

if [ $# -eq 0 ]; then
  echo "用法: bash summary_metric.sh <日志文件> [另一个日志文件...]"
  echo
  echo "例如（基线与优化对比）:"
  echo "  bash summary_metric.sh \\"
  echo "    /workspace/ascend_ws/MindSpeed-MM/logs/train_eager_200.log \\"
  echo "    /workspace/ascend_ws/MindSpeed-MM/logs/train_ascendc_200.log"
  exit 1
fi

step_time_of() {
  grep "elapsed time per iteration" "$1" 2>/dev/null \
    | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' \
    | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 \
    | awk '{s+=$1; n++} END {if (n>0) printf "%.1f", s/n}'
}
gbs_of() {
  grep "global batch size" "$1" 2>/dev/null \
    | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' \
    | head -n 1 | awk '{print $1}'
}
steps_of() {
  grep -c 'elapsed time per iteration' "$1" 2>/dev/null
}

echo "==================================================="
echo " 统一口径性能指标"
echo " 统计窗口 = 第 101-200 步（上游脚本 head -200 | tail -100）"
echo "==================================================="

declare -a NAME_ARR STIME SPEED
i=0
for LOG in "$@"; do
  if [ ! -f "$LOG" ]; then
    echo
    echo "[$(basename "$LOG")]  ✗ 找不到这个日志文件"
    echo "    $LOG"
    continue
  fi
  NAME=$(basename "$LOG" .log)
  n=$(steps_of "$LOG")
  st=$(step_time_of "$LOG")
  gb=$(gbs_of "$LOG")

  if [ -z "$st" ] || [ "$st" = "0.0" ]; then
    echo
    echo "[$NAME]  ✗ 算不出指标（日志里没有 iteration 行）"
    echo "    步数: $n"
    echo "    请用 whats_wrong.sh 排查: $LOG"
    continue
  fi

  sp=$(awk -v g="$gb" -v t="$st" 'BEGIN{ if (t>0) printf "%.3f", g*1000/t }')
  NAME_ARR[$i]="$NAME"; STIME[$i]="$st"; SPEED[$i]="$sp"; i=$((i+1))

  echo
  echo "[$NAME]"
  echo "  完成步数      : $n"
  echo "  STEP_TIME(ms) : $st"
  echo "  GBS           : $gb"
  echo "  samples/s     : $sp"
  if [ "$n" -lt 200 ]; then
    echo "  [WARN] 步数不足 200 —— 第 101-200 步窗口不存在，指标已退化为对全部步取均值"
  fi
done

if [ "$i" -ge 2 ]; then
  echo
  echo "==================================================="
  echo " 对比"
  echo "==================================================="
  echo "  基线   : ${NAME_ARR[0]}"
  echo "           ${STIME[0]} ms  →  ${SPEED[0]} samples/s"
  echo "  优化   : ${NAME_ARR[1]}"
  echo "           ${STIME[1]} ms  →  ${SPEED[1]} samples/s"
  echo
  awk -v a="${STIME[0]}" -v b="${STIME[1]}" \
    'BEGIN{ if (b>0) printf "  加速比 : %.3fx（单步耗时 %.1f ms → %.1f ms，降低 %.1f%%）\n", a/b, a, b, (a-b)/a*100 }'
  echo
  echo "  ↑ 这一组就是性能分析报告的核心结论"
fi
echo
