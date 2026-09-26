#!/usr/bin/env python3
"""
fla_npu 源码补丁（幂等，可重复执行）
============================================================
把旧环境上手工试出来的修复固化下来。新环境 clone 源码后一键应用，
不必再走一遍「编译 → 报错 → 猜 → 改」的循环。

修复清单
--------
P1  FLANpuOpApi.cpp 追加 npu_fast_gelu_custom / _backward 定义
    现象: undefined symbol: _ZN6op_api29npu_fast_gelu_custom_backwardERK2at6TensorES3_
    原因: FLANpuPybind.cpp 里声明并注册了这两个函数，OpApi.cpp 里没有定义

P2  FLANpuOpApi.cpp 三处编译错误（自动，按编译日志定位）
    a) cannot bind non-const lvalue reference of type 'bool&'  -> 插入具名 bool 变量
    b) 'layout_str' was not declared in this scope              -> 删除该行
    c) cannot bind non-const lvalue reference of type 'int&'    -> 插入具名 int 变量
    注意: 编译日志里还有 8 条 'copied_params' is not captured 之类的报错，
          它们全在 torch_npu 头文件里，是上面三处的次生报错，不要动。

用法
----
  python3 patch_fla_npu.py apply <fla_dir>              # 应用全部补丁
  python3 patch_fla_npu.py fix   <fla_dir> <errlog>     # 按编译日志修复
  python3 patch_fla_npu.py check <fla_dir>              # 只检查不改
"""
import os
import re
import sys

MARK_BEGIN = "// ---- ascend patch: npu_fast_gelu_custom ----"
MARK_END = "// ---- end ascend patch ----"

# 幂等判断用的宽松匹配：早期版本写入源码的标记前缀不同（不是 ascend），
# 用正则兼容，避免在已打过补丁的源码上重复插入。
MARK_ANY = re.compile(r"// ---- \w+ patch: npu_fast_gelu_custom ----")

GELU_DEFS = f"""{MARK_BEGIN}
at::Tensor npu_fast_gelu_custom(const at::Tensor &self) {{
    at::Tensor out = at::empty_like(self);
    EXEC_NPU_CMD_EXT(aclnnFastGelu, self, out);
    return out;
}}

at::Tensor npu_fast_gelu_custom_backward(const at::Tensor &grad, const at::Tensor &self) {{
    at::Tensor out = at::empty_like(grad);
    EXEC_NPU_CMD_EXT(aclnnFastGeluBackward, grad, self, out);
    return out;
}}
{MARK_END}
"""

OPAPI_REL = "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp"
PYBIND_REL = "torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuPybind.cpp"


def read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def namespace_close_line(text, ns="op_api"):
    """定位 `namespace op_api {` 的匹配闭合大括号所在行号（0-based）"""
    lines = text.split("\n")
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"\s*namespace\s+%s\s*\{" % re.escape(ns), ln):
            start = i
            break
    if start is None:
        return None
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0 and i > start:
            return i
    return None


# ---------------------------------------------------------------- P1
def patch_gelu_defs(fla_dir, dry=False):
    path = os.path.join(fla_dir, OPAPI_REL)
    if not os.path.isfile(path):
        return "skip", f"找不到 {OPAPI_REL}"

    text = read(path)
    if MARK_ANY.search(text):
        return "done", "P1 已打过（幂等跳过）"

    close = namespace_close_line(text, "op_api")
    if close is None:
        # 兜底：直接追加到文件末尾
        new = text.rstrip("\n") + "\n\n" + GELU_DEFS
    else:
        lines = text.split("\n")
        lines.insert(close, GELU_DEFS.rstrip("\n"))
        new = "\n".join(lines)

    if not dry:
        write(path, new)
    return "done", f"P1 已追加 npu_fast_gelu_custom 定义（插入点：op_api 命名空间内）"


# ---------------------------------------------------------------- P2
BIND_RE = re.compile(
    r"(?P<file>[^\s:]+\.cpp):(?P<line>\d+):(?P<col>\d+):\s*error:\s*"
    r"cannot bind non-const lvalue reference of type\s*'(?P<type>bool|int)\s*&'"
)
NOTDECL_RE = re.compile(
    r"(?P<file>[^\s:]+\.cpp):(?P<line>\d+):(?P<col>\d+):\s*error:\s*"
    r"'(?P<name>\w+)' was not declared in this scope"
)

# 每种类型对应的变量名（与旧环境上试通的一致）
VAR_NAME = {"bool": "use_exp2_flag", "int": "neg_one"}

LITERAL_RE = {
    "bool": re.compile(r"\b(true|false)\b"),
    "int": re.compile(r"(?<![\w.])-?\d+(?![\w.])"),
}


def parse_errors(errlog):
    """从编译日志里提取「源文件自身」的错误（忽略 torch_npu 头文件里的次生报错）"""
    if not os.path.isfile(errlog):
        return []
    text = read(errlog)
    out = []
    for m in BIND_RE.finditer(text):
        if "torch_npu" in m.group("file") or "/include/" in m.group("file"):
            continue
        out.append(("bind", m.group("file"), int(m.group("line")), m.group("type")))
    for m in NOTDECL_RE.finditer(text):
        if "torch_npu" in m.group("file") or "/include/" in m.group("file"):
            continue
        out.append(("notdecl", m.group("file"), int(m.group("line")), m.group("name")))
    # 同一行只修一次
    seen, uniq = set(), []
    for e in out:
        k = (e[1], e[2])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)
    return uniq


def fix_bind(path, lineno, type_name, dry=False):
    """把传给 `T&` 的字面量提成具名变量，插在该行前面"""
    lines = read(path).split("\n")
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return False, f"行号越界: {lineno}"

    var = VAR_NAME[type_name]
    target = lines[idx]

    if var in target:
        return False, f"第 {lineno} 行已含 {var}（幂等跳过）"

    m = LITERAL_RE[type_name].search(target)
    if not m:
        return False, f"第 {lineno} 行找不到 {type_name} 字面量，需手工看: {target.strip()[:80]}"

    literal = m.group(0)
    indent = re.match(r"\s*", target).group(0)
    decl = f"{indent}{type_name} {var} = {literal};"
    newline = target[: m.start()] + var + target[m.end():]

    if not dry:
        lines.insert(idx, decl)
        lines[idx + 1] = newline
        write(path, "\n".join(lines))
    return True, f"第 {lineno} 行: 提取 {type_name} {var} = {literal}"


def fix_notdecl(path, lineno, name, dry=False):
    """删除引用未声明变量的整行"""
    lines = read(path).split("\n")
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return False, f"行号越界: {lineno}"
    removed = lines[idx]
    if not dry:
        del lines[idx]
        write(path, "\n".join(lines))
    return True, f"第 {lineno} 行已删除（引用未声明的 {name}）: {removed.strip()[:70]}"


def fix_from_log(fla_dir, errlog, dry=False):
    errs = parse_errors(errlog)
    if not errs:
        return 0, ["未从日志解析到源文件自身的错误"]

    msgs, n = [], 0
    # 从后往前修，避免行号漂移
    for kind, f, lineno, payload in sorted(errs, key=lambda e: -e[2]):
        path = os.path.join(fla_dir, OPAPI_REL)
        if not os.path.isfile(path):
            msgs.append(f"找不到 {OPAPI_REL}")
            continue
        if kind == "bind":
            ok_, m = fix_bind(path, lineno, payload, dry)
        else:
            ok_, m = fix_notdecl(path, lineno, payload, dry)
        msgs.append(("  ✓ " if ok_ else "  - ") + m)
        if ok_:
            n += 1
    return n, msgs


# ---------------------------------------------------------------- 入口
def apply_all(fla_dir, dry=False):
    print("=== fla_npu 源码补丁 ===")
    st, msg = patch_gelu_defs(fla_dir, dry)
    print(("  ✓ " if st == "done" else "  - ") + msg)

    pybind = os.path.join(fla_dir, PYBIND_REL)
    if os.path.isfile(pybind):
        t = read(pybind)
        if "npu_fast_gelu_custom" in t:
            print("  ✓ 确认 FLANpuPybind.cpp 已声明 npu_fast_gelu_custom（无需改）")
        else:
            print("  ! FLANpuPybind.cpp 里没找到 npu_fast_gelu_custom，P1 可能不需要")
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd, fla_dir = sys.argv[1], sys.argv[2]
    dry = os.environ.get("DRY_RUN") == "1"

    if not os.path.isdir(fla_dir):
        print(f"[ERROR] 不是目录: {fla_dir}")
        return 1

    if cmd == "apply":
        return apply_all(fla_dir, dry)
    if cmd == "fix":
        if len(sys.argv) < 4:
            print("用法: patch_fla_npu.py fix <fla_dir> <errlog>")
            return 1
        n, msgs = fix_from_log(fla_dir, sys.argv[3], dry)
        print("=== 按编译日志修复 ===")
        for m in msgs:
            print(m)
        print(f"共修复 {n} 处")
        return 0
    if cmd == "check":
        return apply_all(fla_dir, dry=True)

    print(f"[ERROR] 未知命令: {cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
