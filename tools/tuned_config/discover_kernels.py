# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static discovery of ``@triton.jit`` kernels in the vLLM package (B3).

AST-walks the source tree (no imports -- robust to torch/CUDA/platform-gated
import failures) and returns the set of functions decorated with
``@triton.jit`` (matched as the attribute access ``triton.jit`` or a bare
``jit`` imported from triton).

Each kernel is identified by ``"<module.path>:<function_name>"`` so that two
kernels with the same simple name in different modules do not collide. This set
is the left-hand side of the B3 completeness ratchet: discovered kernels must be
either registered tunable or allowlisted with a reason.
"""

import argparse
import ast
import os
from collections.abc import Iterable


def _module_path(py_file: str, root_dir: str, top_package: str) -> str:
    """Turn a file path into a dotted module path rooted at ``top_package``.

    e.g. <root>/model_executor/layers/x.py -> vllm.model_executor.layers.x
    """
    rel = os.path.relpath(py_file, root_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    parts = rel_no_ext.split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([top_package, *parts]) if parts else top_package


def _is_triton_jit(decorator: ast.expr) -> bool:
    """True if an AST decorator node is ``triton.jit`` / ``@triton.jit(...)`` or
    a bare ``jit`` / ``jit(...)`` (jit imported directly from triton)."""
    # Unwrap a call: @triton.jit(...) -> decorator is a Call whose .func is the
    # thing we match.
    node = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(node, ast.Attribute):
        # triton.jit  (or something.triton.jit)
        return node.attr == "jit" and (
            isinstance(node.value, ast.Name) and node.value.id == "triton"
        )
    if isinstance(node, ast.Name):
        # bare `jit` brought in via `from triton import jit`
        return node.id == "jit"
    return False


def _kernels_in_file(py_file: str, module_path: str) -> Iterable[str]:
    with open(py_file, encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=py_file)
        except SyntaxError:
            return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_is_triton_jit(d) for d in node.decorator_list):
                found.append(f"{module_path}:{node.name}")
    return found


def discover_triton_kernels(root: str = "vllm") -> set[str]:
    """Return the set of ``"<module.path>:<func>"`` for every ``@triton.jit``
    function found by static AST analysis under ``root``.

    ``root`` may be the package name ``"vllm"`` (resolved relative to this
    file's repo) or an absolute path to the package directory.
    """
    if os.path.isabs(root):
        root_dir = root
        top_package = os.path.basename(root.rstrip(os.sep))
    else:
        # tools/tuned_config/discover_kernels.py -> repo root is two dirs up.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        root_dir = os.path.join(repo_root, root)
        top_package = root

    kernels: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            py_file = os.path.join(dirpath, name)
            module_path = _module_path(py_file, root_dir, top_package)
            kernels.update(_kernels_in_file(py_file, module_path))
    return kernels


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="vllm", help="package name or abs path")
    parser.add_argument(
        "--count", action="store_true", help="print only the count"
    )
    args = parser.parse_args()
    kernels = discover_triton_kernels(args.root)
    if args.count:
        print(len(kernels))
    else:
        for k in sorted(kernels):
            print(k)


if __name__ == "__main__":
    _main()
