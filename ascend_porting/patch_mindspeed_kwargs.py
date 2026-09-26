#!/usr/bin/env python3
"""
MindSpeed-MM 传参源码级补丁（AST 精确定位，幂等）
============================================================
问题
----
MindSpeed-MM 的 Qwen3.5 GDN 层按上游 flash-linear-attention 的签名调用
torch.ops.npu 算子，会多带三个关键字：
  save_new_value / use_exp2 / transpose_state_layout
而 fla_npu 的算子 schema 里没有这三个形参，运行期直接崩：
  TypeError: npu_recompute_w_u_fwd() got an unexpected keyword argument 'save_new_value'

为什么要在源码层改
------------------
运行期垫片（patches/npu_ops_compat.py）能兜住，但那是「事后补救」：
每次调用都要先撞一次 TypeError 再重试。源码层改掉是「事前消除」，
确定性更高、没有额外开销。两层都装上，互为保险。

为什么用 AST 而不是正则
-----------------------
正则无法区分「函数调用实参」和「赋值 / 形参默认值 / 字典键」：
  self.save_new_value = True          <- 赋值，绝不能删
  def f(save_new_value=True):         <- 形参默认值，不能删
  op(a, save_new_value=True)          <- 调用实参，要删
AST 能精确判定：只删 ast.Call 节点里的关键字实参。

作用范围（保守）
----------------
只处理 <repo>/mindspeed_mm/ 下、且内容里出现 torch.ops.npu / fla_npu /
npu_recompute 之一的 .py 文件。

用法
----
  python3 patch_mindspeed_kwargs.py <repo_dir>            # 打补丁
  python3 patch_mindspeed_kwargs.py <repo_dir> --check    # 只看会改什么
"""
import ast
import os
import sys

NAMES = ("save_new_value", "use_exp2", "transpose_state_layout")
HINT_TOKENS = ("torch.ops.npu", "fla_npu", "npu_recompute")
SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".venv", "site-packages"}


def read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def candidates(repo):
    root = os.path.join(repo, "mindspeed_mm")
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                text = read(p)
            except Exception:
                continue
            if any(t in text for t in HINT_TOKENS):
                out.append((p, text))
    return out


def _line_offsets(text):
    offs = [0]
    for line in text.splitlines(keepends=True):
        offs.append(offs[-1] + len(line))
    return offs


def _abs(offs, lineno, col):
    return offs[lineno - 1] + col


def find_spans(text):
    """返回 (要删除的区间列表, 命中的关键字名集合, 解析错误)"""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [], set(), e

    offs = _line_offsets(text)
    spans = []
    hit = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in NAMES:
                continue
            hit.add(kw.arg)
            start = _abs(offs, kw.lineno, kw.col_offset)
            end = _abs(offs, kw.end_lineno, kw.end_col_offset)

            # 优先吞掉后面的逗号，否则吞掉前面的逗号
            j = end
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] == ",":
                end = j + 1
            else:
                i = start
                while i > 0 and text[i - 1] in " \t\r\n":
                    i -= 1
                if i > 0 and text[i - 1] == ",":
                    start = i - 1
            spans.append((start, end))

    spans.sort()
    # 去重（嵌套时可能重复）
    uniq = []
    for s, e in spans:
        if uniq and s <= uniq[-1][1]:
            uniq[-1] = (uniq[-1][0], max(uniq[-1][1], e))
        else:
            uniq.append((s, e))
    return uniq, hit, None


def apply_spans(text, spans):
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + text[e:]
    return text


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    repo = sys.argv[1]
    dry = "--check" in sys.argv

    if not os.path.isdir(repo):
        print("[ERROR] 不是目录: %s" % repo)
        return 1

    print("=== MindSpeed-MM 传参补丁（AST）===")
    print("  仓库: %s" % repo)

    files = candidates(repo)
    if not files:
        print("  [WARN] mindspeed_mm/ 下没找到引用 npu 算子的文件 ——")
        print("         可能仓库没拉全，或上游改了写法。此时靠运行期垫片兜底。")
        return 0

    print("  候选文件 %d 个" % len(files))
    total = 0
    bad = 0
    for path, text in files:
        rel = os.path.relpath(path, repo)
        spans, hit, err = find_spans(text)
        if err is not None:
            bad += 1
            print("  [跳过] %s：语法解析失败（%s），不猜着改" % (rel, err.msg))
            continue
        if not spans:
            continue
        new = apply_spans(text, spans)
        try:
            ast.parse(new)
        except SyntaxError as e:
            print("  [跳过] %s：改完语法坏了（%s），回滚不改" % (rel, e.msg))
            bad += 1
            continue
        total += len(spans)
        if not dry:
            write(path, new)
        print("  ✓ %s：删除 %d 处  [%s]" % (rel, len(spans), ", ".join(sorted(hit))))

    print()
    if dry:
        print("  （--check，未写入）共会删除 %d 处" % total)
    else:
        print("  共删除 %d 处多余关键字" % total)
        if total == 0 and bad == 0:
            print("  → 源码里没有这三个关键字，说明上游已改或走的是动态 kwargs；")
            print("    运行期垫片会兜住，不必担心。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
