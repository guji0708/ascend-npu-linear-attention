#!/bin/bash
# 安装 fla_npu 算子库（ascendc 优化轮的前置条件）
# 用法: bash install_fla_npu.sh
# 若 GitHub 不通，可指定镜像: REPO_URL=<镜像地址> bash install_fla_npu.sh
# 全程日志: /workspace/c4ai/fla_npu_install.log
set -u
PY=/usr/local/python3.11.15/bin/python
WORK=/workspace/c4ai
REPO_URL="${REPO_URL:-https://github.com/flashserve/flash-linear-attention-npu.git}"
SRC_DIR="$WORK/flash-linear-attention-npu"
LOG="$WORK/fla_npu_install.log"

echo "==================================================="
echo " 安装 fla_npu（AscendC 算子库）"
echo " 全程日志: $LOG"
echo "==================================================="

# 从这一刻起，所有输出同时进日志和屏幕
exec > >(tee -a "$LOG") 2>&1

if $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('fla_npu') else 1)"; then
  echo "→ fla_npu 已安装，无需重复安装"
  exit 0
fi

echo
echo "=== 0) 环境基线（出问题时先看这一段）==="
echo "  ASCEND_HOME_PATH : ${ASCEND_HOME_PATH:-<未设置>}"
echo "  CANN set_env.sh  : $(ls /usr/local/Ascend/cann/set_env.sh 2>&1 | tail -1)"
echo "  python           : $($PY -V 2>&1)"
echo "  gcc              : $(command -v gcc || echo 缺失)"
echo "  make             : $(command -v make || echo 缺失)"
echo "  git              : $(command -v git || echo 缺失)"
echo "  torch_npu        : $($PY -c 'import torch_npu; print(torch_npu.__version__)' 2>&1 | tail -1)"

echo
echo "=== 1) 加载 CANN 环境 ==="
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  source /usr/local/Ascend/cann/set_env.sh
  echo "  已 source，ASCEND_HOME_PATH 现在是: ${ASCEND_HOME_PATH:-<仍为空！这就有问题>}"
else
  echo "  [ERROR] 找不到 /usr/local/Ascend/cann/set_env.sh，停下，告诉队长"
  exit 1
fi

echo
echo "=== 2) 探测芯片类型（决定 --soc）==="
CHIP=$($PY -c "import torch_npu; print(torch_npu.npu.get_device_name(0))" 2>&1 | tail -1)
echo "  get_device_name(0) 返回: $CHIP"
case "$CHIP" in
  *9382*|*93*)   SOC=ascend910_93 ;;
  *910B*|*910b*) SOC=ascend910b ;;
  *) echo "  [WARN] 芯片名未识别，默认用 ascend910_93"
     echo "         （若编译报 SoC 不识别，把本脚本里的 ascend910_93 改成 ascend910b 重试）"
     SOC=ascend910_93 ;;
esac
echo "  选用 --soc=$SOC"

echo
echo "=== 3) 拉取源码（先探网络，不闷着失败）==="
cd "$WORK" || exit 1
if [ -d "$SRC_DIR" ]; then
  echo "  源码目录已存在，跳过 clone"
else
  echo "  探测 $REPO_URL （最多等 25 秒）..."
  if timeout 25 git ls-remote "$REPO_URL" HEAD > /dev/null 2>&1; then
    echo "  可达，开始 clone"
    git clone "$REPO_URL" "$SRC_DIR" || { echo "  [ERROR] clone 失败"; exit 1; }
  else
    echo "  [ERROR] 源码仓库不可达（git ls-remote 25 秒无响应）"
    echo
    echo "  这台机器大概率连不上 GitHub。三种出路："
    echo "    a) 若有 gitee 镜像，用环境变量覆盖："
    echo "       REPO_URL=<镜像地址> bash install_fla_npu.sh"
    echo "    b) 在能上网的机器上 git clone 后打包，传进来解压到 $SRC_DIR"
    echo "    c) 放弃 ascendc 优化轮，用现有 eager 基线数据写报告，并说明受阻原因"
    echo
    echo "  请把这一段发给队长"
    exit 1
  fi
fi
cd "$SRC_DIR" || exit 1
git checkout c2e3d83f 2>/dev/null || echo "  [WARN] 切 commit c2e3d83f 失败，继续用当前版本"
echo "  当前 commit: $(git rev-parse --short HEAD 2>/dev/null)"

echo
echo "=== 4) 编译算子 run 包（最久的一步，输出落盘不吞错）==="
BUILD_LOG="$WORK/fla_npu_build.log"
bash build.sh --soc="$SOC" --pkg --vendor_name=fla_npu > "$BUILD_LOG" 2>&1
rc=$?
echo "  返回码: $rc"
echo "  完整日志: $BUILD_LOG"
echo "  --- 日志末尾 15 行 ---"
tail -15 "$BUILD_LOG"

runpkg=$(ls build_out/fla-npu_*.run 2>/dev/null | head -1)
if [ $rc -ne 0 ] || [ -z "$runpkg" ]; then
  echo
  echo "  [ERROR] 编译失败。真正的错误行如下（整段发给队长）："
  echo "  ===== 错误关键行 ====="
  grep -n -iE "error|not found|no such file|undefined|fatal|cannot find|No such" "$BUILD_LOG" | head -30
  echo "  ====================="
  exit 1
fi
echo "  已生成: $runpkg"

echo
echo "=== 5) 安装 run 包 ==="
bash "$runpkg" 2>&1 | tail -15

echo
echo "=== 6) 编译 torch 自定义算子 ==="
cd torch_custom/fla_npu/ || { echo "  [ERROR] 找不到 torch_custom/fla_npu/"; exit 1; }
TORCH_LOG="$WORK/fla_npu_torch_build.log"
bash build.sh > "$TORCH_LOG" 2>&1
rc2=$?
echo "  返回码: $rc2"
echo "  完整日志: $TORCH_LOG"
tail -15 "$TORCH_LOG"
if [ $rc2 -ne 0 ]; then
  echo
  echo "  [ERROR] torch 算子编译失败，关键行如下（整段发给队长）："
  grep -n -iE "error|not found|undefined|fatal|cannot find" "$TORCH_LOG" | head -30
  exit 1
fi

echo
echo "=== 7) 验收 ==="
if $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('fla_npu') else 1)"; then
  echo "  ✓ 安装成功，fla_npu 可用"
  $PY -m pip list 2>/dev/null | grep -i fla
  echo
  echo "→ 下一步: bash scripts/run_round.sh ascendc 200"
else
  echo "  ✗ 仍然不可用"
  echo "  请把整份日志发给队长: $LOG"
  exit 1
fi
