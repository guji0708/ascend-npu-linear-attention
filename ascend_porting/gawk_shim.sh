#!/bin/bash
# c4ai-gawk-shim
# ============================================================
# gawk 替身
# ------------------------------------------------------------
# 背景：镜像只装了 mawk，没有 gawk。而 fla_npu 的 build.sh 里
#       用 gawk 的 strftime() 扩展生成版本号时间戳，直接跑会
#         build.sh: line 1933: gawk: command not found   (exit 127)
#       不能用 apt 装 gawk（apt 依赖树已损坏，修它会毁容器）。
#
# 做法：拦下 awk 程序体里的 strftime("fmt") 调用，用 Python 先算出
#       时间字符串替换成字面量，再交给 mawk 执行。其余情况原样转发。
#
# 安装位置：/usr/local/bin/gawk（PATH 优先级高于 /usr/bin）
# ============================================================
set -u

PY_BIN="${PY:-/usr/local/python3.11.15/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN="$(command -v python3)"

declare -a A=("$@")

for i in "${!A[@]}"; do
  case "${A[$i]}" in
    *strftime*)
      A[$i]=$(printf '%s' "${A[$i]}" | "$PY_BIN" -c '
import sys, re, time
prog = sys.stdin.read()
def sub(m):
    return "\"" + time.strftime(m.group(1)) + "\""
sys.stdout.write(re.sub(r"strftime\(\s*\"([^\"]*)\"\s*\)", sub, prog))
')
      ;;
  esac
done

exec mawk "${A[@]}"
