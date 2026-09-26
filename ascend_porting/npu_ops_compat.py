#!/usr/bin/env python3
"""
fla_npu 参数兼容垫片（可自动加载，与算子注册顺序无关）
============================================================
问题
----
MindSpeed-MM 按上游 flash-linear-attention 的签名传参，会带
  save_new_value / use_exp2 / transpose_state_layout
这三个关键字；而 fla_npu 的算子包装层没有这三个形参（内部取固定值），
直接调用会崩：
  TypeError: npu_recompute_w_u_fwd() got an unexpected keyword argument 'save_new_value'

做法
----
包装 torch.ops.npu 里的算子，调用前把这几个多余关键字丢掉。
关键点：包装挂在 `_OpNamespace.__getattr__` 上，是**懒执行**的，
所以不依赖「垫片先于 fla_npu 注册算子」——谁先谁后都能覆盖到。

为什么必须静默
--------------
每类参数只提示一次（首次提示，之后静默）。刷屏会干扰性能数据测量和进度监控，
旧环境就吃过这个亏。

三种加载方式（任一即可）
------------------------
1) 自动（推荐）：本文件被复制到 site-packages，并配一个 .pth：
       c4ai_fla_compat.pth  内容: import npu_ops_compat
   再设环境变量 FLA_NPU_COMPAT=1，则所有 Python 进程自动生效。
2) 手动：import npu_ops_compat; npu_ops_compat.install()
3) 直接跑：python3 npu_ops_compat.py --selftest

自检
----
  python3 npu_ops_compat.py --selftest
会验证：补丁是否装上、多余关键字是否真被丢掉。
"""
import os
import sys

HARMLESS = ("save_new_value", "use_exp2", "transpose_state_layout")
MUST_TRUE = {"save_new_value"}

_SEEN = set()
_VERBOSE = True
_PATCHED = False

ENV_FLAG = "FLA_NPU_COMPAT"
ENV_VERBOSE = "FLA_NPU_COMPAT_VERBOSE"


def _log(name, k, v):
    """每类参数只提示一次，之后静默（避免干扰性能数据测量）"""
    key = (name, k)
    if key in _SEEN:
        return
    _SEEN.add(key)
    if not _VERBOSE:
        return
    flag = "丢弃" if k not in MUST_TRUE else "丢弃(该值恒为 True，语义一致)"
    print(
        "[fla_npu_compat] %s: %s %s=%r（首次提示，后续静默）" % (name, flag, k, v),
        file=sys.stderr,
    )


def _accepted(fn):
    """取函数真正接受的参数名集合；取不到返回 None（表示不预裁剪）"""
    try:
        import inspect

        return set(inspect.signature(fn).parameters)
    except Exception:
        return None


def _wrap(name, fn):
    acc = _accepted(fn)

    def wrapper(*args, **kwargs):
        if acc is not None:
            for k in list(kwargs):
                if k in HARMLESS and k not in acc:
                    _log(name, k, kwargs.pop(k))
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            # 兜底：OpOverloadPacket 取不到签名时，靠这条路径剥离
            msg = str(e)
            if not any(k in msg for k in HARMLESS):
                raise
            kw = dict(kwargs)
            for k in HARMLESS:
                if k in kw:
                    _log(name, k, kw.pop(k))
            return fn(*args, **kw)

    wrapper.__name__ = getattr(fn, "__name__", name)
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    wrapper.__wrapped__ = fn
    wrapper.__c4ai_fla_compat__ = name
    return wrapper


def _ascendc_op(name):
    """回落到 fla_npu.ops.ascendc 的 ctypes 包装。

    commit edfae99e 的 ascendc 实现走 aclnn ctypes 直连（libfla_npu_stable.so
    + OPP run 包），不注册 torch.ops.npu；而 legacy torch 扩展要求
    torch_npu>=2.10.0.post2，本机是 2.10.0，setup.py 的构建检查直接拒，编不出来。
    所以 MindSpeed-MM 那句 torch.ops.npu.npu_recompute_w_u_fwd 得靠这里接住。
    """
    try:
        import importlib

        A = importlib.import_module("fla_npu.ops.ascendc")
    except Exception:
        return None
    fn = getattr(A, name, None)
    if fn is None and name.startswith("npu_"):
        fn = getattr(A, name[4:], None)
    return fn if callable(fn) else None


_NPU_NS = None


def _npu_namespace():
    """拿 torch.ops.npu 这个命名空间对象（缓存）。"""
    global _NPU_NS
    if _NPU_NS is None:
        try:
            import torch

            _NPU_NS = torch.ops.npu
        except Exception:
            return None
    return _NPU_NS


def _is_npu_namespace(ns):
    """判断是不是 torch.ops.npu。

    这里**绝不能**写 getattr(ns, "_name", None)：该属性取不到时会重新进入
    __getattr__，直接自递归成 RecursionError（本文件踩过）。
    只用对象身份比较，退化时用 object.__getattribute__ 绕过 __getattr__ 兜底。
    """
    if ns is _npu_namespace():
        return True
    for attr in ("__name__", "_name"):
        try:
            if object.__getattribute__(ns, attr) == "npu":
                return True
        except Exception:
            continue
    return False


def _patch_namespace_class():
    """把包装挂到 _OpNamespace.__getattr__ 上（懒执行，顺序无关）"""
    global _PATCHED
    if _PATCHED:
        return True

    import importlib

    _ops = importlib.import_module("torch._ops")
    NS = getattr(_ops, "_OpNamespace", None)
    if NS is None:
        return False
    if getattr(NS, "_c4ai_fla_compat_patched", False):
        _PATCHED = True
        return True

    orig_getattr = NS.__getattr__

    def __getattr__(self, name):  # noqa: N807
        # 下划线开头一律交还原实现；非 npu 命名空间一律不碰
        # （包住 aten/prim 等命名空间会把 OpOverloadPacket 换成普通函数，
        #   破坏 torch.ops.aten.x.default(...) 这类调用）
        if name.startswith("_") or not _is_npu_namespace(self):
            return orig_getattr(self, name)
        try:
            fn = orig_getattr(self, name)
        except AttributeError:
            # 本版本 ascendc 不注册 torch.ops.npu（走 ctypes），这里接住调用
            fn = _ascendc_op(name)
            if fn is None:
                raise
        if not callable(fn):
            return fn
        w = _wrap(name, fn)
        # 覆盖实例缓存，避免第二次访问绕过包装
        try:
            self.__dict__[name] = w
        except Exception:
            pass
        return w

    NS.__getattr__ = __getattr__
    NS._c4ai_fla_compat_patched = True
    _PATCHED = True
    return True


def install(verbose=True):
    """手动安装（torch 已导入时用这个）"""
    global _VERBOSE
    if os.environ.get(ENV_VERBOSE) == "0":
        verbose = False
    _VERBOSE = verbose
    try:
        ok = _patch_namespace_class()
    except Exception as e:
        print("[fla_npu_compat] 安装失败: %r" % (e,), file=sys.stderr)
        return 0
    if not ok:
        print("[fla_npu_compat] 找不到 torch._ops._OpNamespace，未安装", file=sys.stderr)
        return 0
    # 顺带把 ascendc 包装挂到 torch_npu.ops —— 部分调用点走的是这个命名空间
    try:
        import importlib

        importlib.import_module("fla_npu.ops.ascendc").install_torch_npu_ops_compat()
    except Exception:
        pass
    if verbose:
        print("[fla_npu_compat] 已挂载 torch.ops.npu 参数兼容包装（懒执行）", file=sys.stderr)
    return 1


def is_active():
    try:
        import importlib

        NS = importlib.import_module("torch._ops")._OpNamespace
        return bool(getattr(NS, "_c4ai_fla_compat_patched", False))
    except Exception:
        return False


# ---------------------------------------------------------------- 自动加载
class _TorchFinder:
    """拦截 torch 的 import，在其加载完成后立刻挂上包装。

    这样避免在 site 初始化阶段就 import torch（重且易出问题）。
    """

    _c4ai_fla_compat_finder = True

    def __init__(self):
        self._done = False

    def find_spec(self, fullname, path=None, target=None):
        if self._done or fullname != "torch":
            return None
        self._done = True

        spec = None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None and spec.loader is not None:
                break
            spec = None
        if spec is None or spec.loader is None:
            return None

        inner = spec.loader

        class _Loader:
            def create_module(self, s):
                return inner.create_module(s)

            def exec_module(self, m):
                inner.exec_module(m)
                try:
                    install(verbose=True)
                except Exception as e:
                    print("[fla_npu_compat] 自动挂载失败: %r" % (e,), file=sys.stderr)

            def __getattr__(self, item):
                return getattr(inner, item)

        spec.loader = _Loader()
        return spec


def _boot():
    if os.environ.get(ENV_FLAG) != "1":
        return
    for f in sys.meta_path:
        if getattr(f, "_c4ai_fla_compat_finder", False):
            return
    sys.meta_path.insert(0, _TorchFinder())


_boot()


# ---------------------------------------------------------------- 自检
def _selftest():
    print("=== fla_npu 参数兼容垫片 自检 ===")
    print("  FLA_NPU_COMPAT =", os.environ.get(ENV_FLAG, "<未设置>"))

    import torch

    install(verbose=True)
    if not is_active():
        print("  [FAIL] 补丁未挂载到 torch._ops._OpNamespace")
        return 1
    print("  [OK]   补丁已挂载到 torch._ops._OpNamespace")

    # 用一个假算子验证「多余关键字真被丢掉」
    def _fake_op(a, b=None):
        return (a, b)

    _fake_op.__name__ = "fake_op"
    w = _wrap("fake_op", _fake_op)
    r = w(1, b=2, save_new_value=True, use_exp2=False)
    if r != (1, 2):
        print("  [FAIL] 多余关键字未被丢弃，返回:", r)
        return 1
    print("  [OK]   多余关键字已被丢弃（save_new_value / use_exp2）")

    # 回落到 fla_npu.ops.ascendc 是否生效（本版本 ascendc 不注册 torch.ops.npu）
    for k in ("npu_recompute_w_u_fwd", "npu_fast_gelu_custom"):
        try:
            fn = getattr(torch.ops.npu, k)
            print("  [OK]   torch.ops.npu.%-28s -> %s"
                  % (k, getattr(fn, "__c4ai_fla_compat__", "?")))
        except AttributeError:
            print("  [FAIL] torch.ops.npu.%s 取不到（ascendc 回落未生效）" % k)
            return 1

    # 真实执行一把，确认真能跑在 NPU 上
    try:
        x = torch.randn(4, 8, device="npu", dtype=torch.float16)
        y = torch.ops.npu.npu_fast_gelu_custom(x)
        print("  [OK]   真实执行 npu_fast_gelu_custom ->", tuple(y.shape), y.dtype)
    except Exception as e:
        print("  [FAIL] 真实执行失败:", type(e).__name__, str(e)[:200])
        return 1

    print("  → 自检通过")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    install()
