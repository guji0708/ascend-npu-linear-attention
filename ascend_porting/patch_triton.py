#!/usr/bin/env python3
"""
triton-ascend 与 CANN 9.0.0 兼容补丁（幂等）
============================================================
现象
----
  RuntimeError: Failed to compile .../npu_utils.cpp:321:44:
  error: 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE' is not a member of 'rtLimitType_t'

原因
----
  triton-ascend 在编译 Ascend kernel 前会现场 JIT 编译 npu_utils.cpp，
  用 rtDeviceSetLimit() 设置 WARP_STACK_SIZE / STACK_SIZE / LOW_POWER_TIMEOUT。
  其中 RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE 在 CANN 9.0.0 的 rt_external_base.h
  里不存在（该头文件只定义到 RT_LIMIT_TYPE_STACK_SIZE = 3）。
  即 triton-ascend 3.2.0 期望比 9.0.0 更新的 CANN。

为什么必须修
------------
  MindSpeed-MM 在 ascendc 模式下，causal_conv1d 只支持 triton/ascendc 两档；
  ascendc 轮次会用到 triton 路径，所以 triton 不能是坏的。

做法
----
  删掉所有引用 SIMT_WARP_STACK_SIZE 的初始化项（按内容匹配，不按行号，保证幂等）。
  设备限制少设一项不影响功能，只是少了该项的调优。

用法: python3 patch_triton.py [--dry-run]
"""
import os
import re
import sys

CANDIDATES = [
    "/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp",
]

TOKEN = "SIMT_WARP_STACK_SIZE"


def find_file():
    for p in CANDIDATES:
        if os.path.isfile(p):
            return p
    # 兜底：全盘找
    for root in ("/usr/local/python3.11.15/lib", "/usr/lib/python3", "/usr/local/lib"):
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if fn == "npu_utils.cpp" and "ascend" in dirpath:
                    return os.path.join(dirpath, fn)
    return None


def main():
    dry = "--dry-run" in sys.argv
    path = find_file()
    if not path:
        print("  - 找不到 npu_utils.cpp（triton 未安装或路径不同），跳过")
        return 0

    print(f"  目标: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    hits = [(i, ln) for i, ln in enumerate(lines) if TOKEN in ln]
    if not hits:
        print("  ✓ 已打过补丁（文件中无 SIMT_WARP_STACK_SIZE），幂等跳过")
        return 0

    print(f"  发现 {len(hits)} 行引用 {TOKEN}:")
    for i, ln in hits:
        print(f"    L{i + 1}: {ln.rstrip()[:100]}")

    if dry:
        print("  (dry-run，未写入)")
        return 0

    keep = [ln for ln in lines if TOKEN not in ln]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(keep)
    print(f"  ✓ 已删除 {len(hits)} 行，文件从 {len(lines)} 行变为 {len(keep)} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
