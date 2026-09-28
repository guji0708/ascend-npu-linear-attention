#!/bin/bash
# ============================================================
# 公共库：路径常量、日志、断言、幂等工具
# 所有脚本 source 这个文件，不要各自重复定义
# ============================================================
set -u

# ---------- 固定路径（新环境如与旧环境不同，只改这里） ----------
export WORK=/workspace/ascend_ws
export REPO=$WORK/MindSpeed-MM
export PY=/usr/local/python3.11.15/bin/python
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
export PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LOG_DIR=$WORK/rebuild_logs

# ---------- fla_npu ----------
export FLA_DIR=$WORK/flash-linear-attention-npu
export FLA_URL="${FLA_URL:-https://github.com/flashserve/flash-linear-attention-npu.git}"
# 必须钉在含 stable-ABI + legacy extension 的版本上：
# c2e3d83f（2026-07-10）没有 libfla_npu_stable.so / FLA_NPU_BUILD_LEGACY_EXTENSION / FLANpuPybind.cpp，
# console/install_fla_npu.sh 与 ascend_porting/patch_fla_npu.py 全依赖这三样，钉错了第 3 步必然跑不通。
export FLA_COMMIT="${FLA_COMMIT:-edfae99e}"
export SOC="${SOC:-ascend910_93}"

mkdir -p "$LOG_DIR"

# ---------- 输出 ----------
hr()   { echo "==================================================="; }
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "  [OK]   $*"; }
info() { echo "  [..]   $*"; }
warn() { echo "  [WARN] $*"; }
die()  { echo; echo "  [FATAL] $*"; echo "  → 把上面 30 行完整输出发给队长"; echo; exit 1; }

# ---------- 断言 ----------
need_file() { [ -e "$1" ] || die "缺文件: $1"; }
need_cmd()  { command -v "$1" >/dev/null 2>&1 || die "缺命令: $1"; }
have_py_mod() { $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

# ---------- CANN ----------
source_cann() {
  [ -f "$CANN_ENV" ] || die "找不到 $CANN_ENV"
  # shellcheck disable=SC1090
  source "$CANN_ENV"
  [ -n "${ASCEND_HOME_PATH:-}" ] || die "source 后 ASCEND_HOME_PATH 仍为空"
  ok "CANN 已加载: $ASCEND_HOME_PATH"
}

# ---------- 环境变量（每次新终端都要，写进 bashrc 免踩） ----------
# FLA_NPU_COMPAT=1 会激活参数兼容垫片（见 ascend_porting/npu_ops_compat.py）。
# 没有它，ascendc/triton 轮次会崩在 unexpected keyword argument 'save_new_value'。
ensure_env() {
  export NON_MEGATRON=true
  export FLA_NPU_COMPAT=1
  for kv in \
    'export NON_MEGATRON=true' \
    'export FLA_NPU_COMPAT=1' \
    'export MULTI_STREAM_MEMORY_REUSE=2' \
    'export TASK_QUEUE_ENABLE=2' \
    'export ASCEND_LAUNCH_BLOCKING=0' \
    'export ACLNN_CACHE_LIMIT=100000' \
    'export CPU_AFFINITY_CONF=1' \
    'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True'; do
    grep -qF "$kv" ~/.bashrc 2>/dev/null || echo "$kv" >> ~/.bashrc
  done
  ok "训练环境变量已写入 ~/.bashrc"
}

# ---------- site-packages 路径（装兼容垫片用） ----------
site_packages() {
  $PY -c "import site; print(site.getsitepackages()[0])" 2>/dev/null \
    || echo /usr/local/python3.11.15/lib/python3.11/site-packages
}

# ---------- 训练进程检测 ----------
training_running() {
  ps -eo args 2>/dev/null | grep -q "[m]indspeed_mm/fsdp/train/trainer.py"
}

# ---------- 可用 CPU 核数（给 taskset 用，留 2 核给 WebIDE） ----------
safe_cores() {
  local n
  n=$(nproc)
  if [ "$n" -gt 3 ]; then echo $((n - 2)); else echo 1; fi
}

# ---------- 日志尾部（失败时抓关键行） ----------
tail_err() {
  local f="$1" n="${2:-25}"
  echo "  --- 关键错误行 ---"
  grep -n -iE "error:|RuntimeError|ImportError|ModuleNotFoundError|ValueError|undefined symbol|No such file|cannot find|fatal error" "$f" 2>/dev/null | head -n "$n"
  echo "  --- 末尾 10 行 ---"
  tail -10 "$f" 2>/dev/null
}
