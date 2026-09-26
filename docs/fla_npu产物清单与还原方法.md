# fla_npu 编译产物清单与还原方法

> 用途：把已编译好的 AscendC 算子产物带到**同架构的新环境**，免去 25–40 分钟的重新编译。
> 环境前提一致：CANN 9.0.0、Python 3.11.15、torch / torch_npu 2.10.0、SoC `ascend910_93`。

---

## 一、产物在哪

**不在工作目录里，在 site-packages 里。**

这是实际踩到的一个坑：在 `/workspace` 下 `find` 什么都找不到，因为 `pip install` 之后产物落在 Python 的 site-packages 下：

```
/usr/local/python3.11.15/lib/python3.11/site-packages/fla_npu/
```

全盘搜产物时**必须带上 `/usr/local`**，否则一定漏：

```bash
find / -xdev \( -name "*.whl" -o -name "*.run" -o -name "libfla_npu_stable.so" \) 2>/dev/null
```

---

## 二、包内清单（约 129 MB）

| 条目 | 大小 | 说明 |
|---|---|---|
| `libfla_npu_stable.so` | 748,472 B | AscendC 算子注册库。**最关键的一个文件** |
| `opp/` | 目录 | 内嵌的 OPP 算子包，AscendC 算子二进制本体，129 MB 的大头 |
| `ops/` | 目录 | 算子 Python 封装（含 aclnn ctypes 直连封装） |
| `adapters/` | 目录 | 框架适配层 |
| `install_opp.py` | 4,381 B | OPP 安装脚本 |
| `__init__.py` | 9,319 B | 包入口，导入时触发算子注册 |
| `_guard.py` | 7,502 B | 版本守卫（会检查 torch_npu 版本并打 RuntimeWarning） |
| `_compat.py` | 277 B | 兼容垫片入口 |
| `_build_meta.py` | 62 B | 构建元信息 |
| `kda.py` | 385 B | KDA 算子封装 |

### 两个元信息文件（还原时要对照）

`site-packages/fla_npu_opp_env.pth`（23 B）：

```
import fla_npu_opp_env
```

`site-packages/fla_npu/_build_meta.py`：

```python
"""Generated at wheel build time. Do not edit."""
TIER = 'a2'
```

`TIER = 'a2'` 是昇腾的平台分档，说明这批算子是按 A2（910B / 910_93）编的——**换到别的分档不保证兼容**。

### site-packages 根下相关的 `.pth`

| 文件 | 大小 | 作用 |
|---|---|---|
| `fla_npu_opp_env.pth` | 23 B | 挂载 OPP 环境路径（AscendC 算子被找到的前提） |
| `ascend_fla_compat.pth` | 22 B | 算子传参兼容垫片的自动加载钩子（注意：**没有**前导下划线） |

> 导出时只需要包本体与 `fla_npu_opp_env.pth`；`ascend_fla_compat.pth` 由
> `ascend_porting/npu_ops_compat.py` 的安装流程生成，会随环境搭建自动重建。

### 为什么没有 `.whl` 和 `.run`

`console/install_fla_npu.sh` 在构建前会 `rm -rf dist`，装完后 `dist/` 里的 wheel 不再保留；OPP 的 `.run` 包也没有留在 `build_out/` 里。
**但装进 site-packages 的这份是自包含的**——`libfla_npu_stable.so` + 内嵌 `opp/` 就是全部产物，拿到它等于拿到了一切。

---

## 三、怎么导出（在有产物的机器上执行）

```bash
SP=/usr/local/python3.11.15/lib/python3.11/site-packages

# 导出（约 130 MB，压缩后约 41 MB）
tar czf ~/fla_npu_artifacts.tar.gz -C $SP fla_npu fla_npu_opp_env.pth
echo "tar exit=$?"

# 校验：完整性 + 关键文件在不在 + 条目数
tar tzf ~/fla_npu_artifacts.tar.gz > /dev/null && echo "gzip+tar 完整"
tar tzf ~/fla_npu_artifacts.tar.gz | grep -E "libfla_npu_stable\.so|\.pth$"
tar tzf ~/fla_npu_artifacts.tar.gz | grep -c "fla_npu/opp/"
find $SP/fla_npu -type f | wc -l
```

> **注意**：不要把多个 `-name` 用 `-o` 串起来却不加括号，也不要在 tar 里带不存在的路径——GNU tar 会「继续打包但以非 0 退出」，包看着是好的，但退出码骗人。带上 `echo "tar exit=$?"` 确认是 0。

### 实测基线（可作对照）

| 校验项 | 实测值 |
|---|---|
| `tar exit` | **0** |
| 归档大小 | **42,541,845 B**（41 MB；源 129 MB） |
| 压缩包完整性（`tar tzf > /dev/null`） | 完整 |
| 归档内文件数 | **1374** |
| `fla_npu/opp/` 下条目数 | **1311** |
| 关键文件 | `fla_npu/libfla_npu_stable.so`、`fla_npu_opp_env.pth` 均在 |

> **条目数口径说明**：首次导出时记的是 1145（对应源目录 1144 个文件）；最终归档为 **1374** 个文件，比首次多 229 个——中间补编过 legacy 扩展，多出来的就是那部分。两次数都属正常，**不要拿 1145 当硬判据**。真正的判据是第四节的运行时验收；条目数只作完整性参考，明显偏少（比如不到 1000）才说明包不全。

---

## 四、怎么还原

### 4.1 用脚本（推荐）

```bash
bash console/restore_fla_npu.sh

# 产物包不在默认位置时
FLA_ARTIFACTS=/path/to/fla_npu_artifacts.tar.gz bash console/restore_fla_npu.sh
```

脚本依次做四件事，任一步不过就以非 0 退出（此时改用源码编译）：
① 找包 → ② `tar tzf` 完整性 + `libfla_npu_stable.so` 存在性 → ③ 解包到 site-packages → ④ 运行时验收。

### 4.2 手工还原

```bash
SP=/usr/local/python3.11.15/lib/python3.11/site-packages
tar xzf fla_npu_artifacts.tar.gz -C $SP
```

验收：算子总数应从 **354**（裸基线）涨到包含自定义算子的数量：

```bash
/usr/local/python3.11.15/bin/python - <<'PY'
import torch, torch_npu, fla_npu
n = [x for x in dir(torch.ops.npu) if not x.startswith('_')]
print("torch.ops.npu 算子条目数:", len(n))
print("npu_recompute_w_u_fwd 存在:", hasattr(torch.ops.npu, "npu_recompute_w_u_fwd"))
PY
```

判据：

- 裸环境（一个自定义算子都没注册）时算子总数为 **354**，且没有 `npu_recompute_w_u_fwd`；
- 还原成功后该算子应存在，算子总数应高于 354。

> 导入时会打一条 `RuntimeWarning: fla_npu: detected torch_npu 2.10.0, below the recommended minimum 2.10.0.post2`。
> 这是**已知且预期**的——本环境正是 2.10.0，走的是 Stable-ABI + 运行期 ctypes 回落路线，警告本身不影响算子可用性。

---

## 五、还原不成功怎么办：三级降级

从快到慢，前一级不通就走下一级：

| 级别 | 路径 | 耗时 | 依赖 |
|---|---|---|---|
| 1 | 还原 `fla_npu_artifacts.tar.gz` | **约 1 分钟** | 已有产物包 |
| 2 | 离线源码 + 编译：把源码包解压到源码目录再跑 `console/install_fla_npu.sh` | 25–40 分钟 | 已有源码包，**不需要联网** |
| 3 | 在线 clone + 编译 | 25–40 分钟 | 能访问上游仓库 |

第 2 级最常用（NPU 机器通常无法直连 GitHub）：

```bash
mkdir -p /workspace/ascend_ws/flash-linear-attention-npu
tar xzf fla_npu_src.tar.gz -C /workspace/ascend_ws/flash-linear-attention-npu --strip-components=1
cd /workspace/ascend_ws/flash-linear-attention-npu
git init -q && git add -A && git commit -qm src
bash /path/to/repo/console/install_fla_npu.sh
```

> 源码补丁由 `ascend_porting/patch_fla_npu.py` 幂等应用（修 `FLANpuOpApi.cpp` 缺失的
> `npu_fast_gelu_custom` / `_backward` 定义，以及三处编译错误），可用
> `python ascend_porting/patch_fla_npu.py apply <源码目录>` 单独调用。
