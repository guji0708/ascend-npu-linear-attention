#!/bin/bash
# ============================================================
# 环境重建脚本
# ------------------------------------------------------------
# 万一计算环境被重建 / 丢失，用这个从零恢复软件环境。
#
# 重要：本脚本固化的是「今天实际跑通的那条路径」，不是官方文档的路径。
#       官方 `scripts/install.sh` 在本镜像上无法直接跑完（原因见第 3 步说明），
#       所以这里把它拆开走。所有细节依据仓库内 docs/环境适配问题与解法.md。
#
# 用法: bash rebuild_env.sh
# 注意: 本脚本只重建软件环境。权重与数据集需另行恢复（见 docs/复现指南.md）。
# ============================================================
set -u

WORK="${ASCEND_WORK:-/workspace/ascend_ws}"
REPO=$WORK/MindSpeed-MM
PY=/usr/local/python3.11.15/bin/python
PIP_MIRROR=https://repo.huaweicloud.com/repository/pypi/simple/

echo "==================================================="
echo " 环境重建（依据：实际验证通过的路径）"
echo "==================================================="

echo
echo "=== 1) 前置确认 ==="
[ -x "$PY" ] || { echo "  [ERROR] 找不到 $PY，停止"; exit 1; }
echo "  Python: $($PY --version 2>&1)"
# 插件式 FSDP2 不走 Megatron 桥接，必须设这个变量（官方启动脚本自带）
export NON_MEGATRON=true
grep -q NON_MEGATRON ~/.bashrc || echo 'export NON_MEGATRON=true' >> ~/.bashrc
echo "  NON_MEGATRON=true 已设（并写入 ~/.bashrc）"

echo
echo "=== 2) pip 镜像换成华为云（清华源会返回 403）==="
$PY -m pip config set global.index-url "$PIP_MIRROR" >/dev/null 2>&1 \
  && echo "  已设置: $PIP_MIRROR" || echo "  [WARN] 设置失败"

echo
echo "=== 3) 拉取 MindSpeed-MM ==="
mkdir -p "$WORK"
if [ -d "$REPO/.git" ]; then
  echo "  已存在，跳过克隆"
else
  git clone https://gitcode.com/Ascend/MindSpeed-MM.git "$REPO" 2>/dev/null \
    || { echo "  gitcode 失败，回退 gitee"; git clone https://gitee.com/ascend/mindspeed-mm.git "$REPO"; } \
    || { echo "  [ERROR] 克隆失败，检查网络"; exit 1; }
fi
cd "$REPO" || exit 1

echo
echo "=== 4) 跑官方 install.sh（它会失败，且这是预期的）==="
echo "  说明: install.sh 会用 pip install -e . 装 MindSpeed-MM 本体，"
echo "        而 pyproject 声明 requires-python >=3.12 且钉 transformers==4.57.0，"
echo "        在本镜像（Python 3.11.15）上必然失败。"
echo "        但它的 --msid 步骤会先成功装上 MindSpeed（仓库内 MindSpeed/ 目录），"
echo "        所以照跑，忽略结尾的失败。"
echo "  另: 必须带 --no，否则无 stdin 时会陷入死循环（曾刷出 3.8GB 日志）"
timeout 1500 bash scripts/install.sh --msid eb10b92 --no > "$WORK/install.log" 2>&1
echo "  install.sh 返回码 = $?（非 0 属预期）"
tail -6 "$WORK/install.log" | sed 's/^/    /'

echo
echo "=== 5) 放宽 Python 版本门禁 ==="
echo "  镜像只有 3.10 / 3.11.15；不能升到 3.12，否则 torch_npu 会失效"
cp -n pyproject.toml pyproject.toml.bak
sed -i 's|requires-python = ">=3.12,<3.13"|requires-python = ">=3.11,<3.13"|' pyproject.toml
grep -n "requires-python" pyproject.toml | sed 's/^/  /'

echo
echo "=== 6) 约束文件：锁死 transformers / torch / torch_npu / numpy ==="
T=$($PY -c "import torch;print(torch.__version__)" 2>/dev/null || echo "?")
N=$($PY -c "import torch_npu;print(torch_npu.__version__)" 2>/dev/null || echo "?")
U=$($PY -c "import numpy;print(numpy.__version__)" 2>/dev/null || echo "?")
printf 'transformers==5.2.0\ntorch==%s\ntorch_npu==%s\nnumpy==%s\n' "$T" "$N" "$U" > /tmp/ascend_constraints.txt
cat /tmp/ascend_constraints.txt | sed 's/^/  /'

echo
echo "=== 7) 按名单补装依赖 ==="
echo "  不要用 pip install -e .（transformers 4.57.0 vs 官方要求 5.2.0 死锁）"
echo "  jsonargparse 必须带 [signatures] extra，否则 mm-convert 报 docstring-parser 缺失"
$PY -m pip install -c /tmp/ascend_constraints.txt \
  accelerate datasets ftfy "jsonargparse[signatures]" numba peft pydantic \
  qwen_vl_utils torchdata megatron-core diffusers modelscope 2>&1 | tail -5

echo
echo "=== 8) 以 --no-deps 安装 MindSpeed-MM 本体 ==="
$PY -m pip install -e . --no-deps 2>&1 | tail -4

echo
echo "=== 9) 补充组件 ==="
bash examples/qwen3_5/install_extensions.sh 2>&1 | tail -4
$PY -m pip install triton-ascend==3.2.0 2>&1 | tail -3

echo
echo "=== 10) 仓库内软链接（配置里用的是相对路径）==="
for d in ckpt dataset; do
  if [ -d "$d" ] && [ ! -L "$d" ]; then
    if [ -z "$(ls -A "$d")" ]; then rmdir "$d"; echo "  已删空目录 $d"; fi
  fi
  if [ ! -e "$d" ]; then
    if [ -e "$WORK/$d" ]; then ln -s "$WORK/$d" "$d" && echo "  已建软链接 $d -> $WORK/$d"
    else echo "  [WARN] $WORK/$d 不存在（权重/数据集还没恢复？）"; fi
  fi
  ls -ld "$d" 2>/dev/null | sed 's/^/  /'
done

echo
echo "=== 11) 验收 ==="
$PY -c "import mindspeed_mm; print('  ✓ import mindspeed_mm OK')" 2>&1 | tail -3
if mm-convert --help >/dev/null 2>&1; then
  echo "  ✓ mm-convert 可用"
else
  echo "  ✗ mm-convert 不可用"
fi
$PY -c "import importlib.util; print('  fla_npu:', 'installed' if importlib.util.find_spec('fla_npu') else 'NOT installed（ascendc 优化轮前需装）')"

echo
echo "==================================================="
echo " 软件环境重建完毕"
echo "==================================================="
echo
echo "接下来还要恢复资产（本脚本不代做）："
echo "  1. HF 权重   -> /workspace/ascend_ws/ckpt/hf_path/Qwen3.5-0.8B   （1.7G）"
echo "  2. DCP 权重  -> mm-convert Qwen35Converter hf_to_dcp"
echo "  3. 数据集    -> /workspace/ascend_ws/dataset                    （316M）"
echo "  详见 docs/复现指南.md"
