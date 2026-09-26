#!/bin/bash
# ============================================================
# 还原预编译的 fla_npu 产物（免去 25-40 分钟重新编译）
# ------------------------------------------------------------
# 背景
#   fla_npu 的 AscendC 算子在 CANN 9.0.0 上完整编译一次要 25-40 分钟。
#   但产物是自包含的：site-packages 下的 fla_npu/（含 libfla_npu_stable.so
#   与内嵌 opp/）+ fla_npu_opp_env.pth 就是全部东西。
#   把这两样打包带到同架构（A2 / A3，SoC ascend910_93）的新环境解包，
#   算子即可直接注册，不必重编译。产物打包方法见
#   docs/fla_npu产物清单与还原方法.md。
#
# 用法
#   bash console/restore_fla_npu.sh
#   FLA_ARTIFACTS=/path/to/fla_npu_artifacts.tar.gz bash console/restore_fla_npu.sh
#
# 退出码
#   0 还原成功且算子验收通过
#   2 没找到产物包        3 产物包不完整
#   4 解包失败            5 算子验收未通过
#   以上非 0 都表示「请改用源码编译」：bash console/install_fla_npu.sh
# ============================================================
set -u

# site-packages 与 Python 解释器（可用环境变量覆盖）
PY="${PY:-/usr/local/python3.11.15/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python || true)"

WORK="${C4AI_WORK:-/workspace/c4ai}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ok()   { echo "  [OK]   $*"; }
info() { echo "  [..]   $*"; }
warn() { echo "  [WARN] $*"; }

echo
echo "=== 还原 fla_npu 预编译产物 ==="

if [ -z "$PY" ]; then
  warn "找不到可用的 python 解释器，请用 PY=/path/to/python 指定"
  exit 4
fi

# ---------------------------------------------------------------- 1) 找包
find_artifacts() {
  local c
  for c in \
    "${FLA_ARTIFACTS:-}" \
    "$WORK/fla_npu_artifacts.tar.gz" \
    "$SELF_DIR/../fla_npu_artifacts.tar.gz" \
    "$HOME/fla_npu_artifacts.tar.gz"; do
    if [ -n "$c" ] && [ -f "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

ART="$(find_artifacts)" || {
  warn "没找到 fla_npu_artifacts.tar.gz"
  echo "        把它放到 $WORK/ 下，或用 FLA_ARTIFACTS=<路径> 指定；"
  echo "        没有产物包就改用源码编译: bash console/install_fla_npu.sh"
  exit 2
}
info "产物包: $ART（$(du -h "$ART" 2>/dev/null | cut -f1)）"

# ---------------------------------------------------------------- 2) 完整性
if ! tar tzf "$ART" > /dev/null 2>&1; then
  warn "产物包不完整（gzip/tar 校验失败），请重新获取，不要用这个包"
  exit 3
fi
if ! tar tzf "$ART" | grep -q "libfla_npu_stable.so"; then
  warn "包内没有 libfla_npu_stable.so —— 不是完整产物包"
  exit 3
fi
info "包内条目数: $(tar tzf "$ART" | wc -l)（完整导出基线约 1374）"
ok "产物包完整"

# ---------------------------------------------------------------- 3) 解包
SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
if [ -z "$SP" ]; then
  SP=/usr/local/python3.11.15/lib/python3.11/site-packages
  warn "取不到 site-packages，回落默认值: $SP"
fi
[ -d "$SP" ] || { warn "site-packages 不存在: $SP"; exit 4; }

info "解包到 $SP ..."
if ! tar xzf "$ART" -C "$SP"; then
  warn "解包失败"
  exit 4
fi
ok "已解包（fla_npu/ + fla_npu_opp_env.pth）"

# ---------------------------------------------------------------- 4) 验收
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann/set_env.sh
else
  warn "没找到 CANN set_env.sh，若下面报 torch_npu 导入失败请先加载 CANN 环境"
fi

"$PY" - <<'PYEOF'
import sys
import torch
try:
    import torch_npu  # noqa: F401
except Exception as e:
    print("  torch_npu 导入失败:", e)
    sys.exit(1)


def _ops():
    return [n for n in dir(torch.ops.npu) if not n.startswith("_")]


# 354 = 一个自定义算子都没注册时的基准值
print("  裸 torch.ops.npu 算子总数:", len(_ops()))

try:
    import fla_npu  # noqa: F401
    import fla_npu.ops.ascendc  # noqa: F401
except Exception as e:
    print("  import fla_npu.ops.ascendc 失败:", e)
    sys.exit(1)

names = _ops()
print("  还原后算子总数:", len(names))
key = ["npu_recompute_w_u_fwd", "npu_fast_gelu_custom", "npu_fast_gelu_custom_backward"]
for k in key:
    print("   ", k.ljust(32), "有" if k in names else "无")
missing = [k for k in key if k not in names]
sys.exit(1 if missing else 0)
PYEOF
rc=$?

if [ $rc -ne 0 ]; then
  warn "算子验收未通过（关键算子缺失）—— 请改用源码编译: bash console/install_fla_npu.sh"
  exit 5
fi

ok "还原成功：三个关键算子都已注册"
echo
echo "  产物与产出本仓库实测数据的那一次同源，复现指标应与之接近。"
echo "  下一步: bash console/health_check.sh"
exit 0
