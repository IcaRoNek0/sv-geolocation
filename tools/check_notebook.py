#!/usr/bin/env python
"""校验 colab.ipynb 的每一格能否编译。生成后必须跑，通过才算改完。

存在的理由：notebook 是生成的（见 make_notebook.py），而生成脚本里的
转义容易差一层——`\\n` 写成 `\n` 会在单元格里变成真实换行，把字符串劈成
两半。这类错误在 notebook 里表现为一句与真实原因毫不相干的语法错误，
而且只有在目标环境里运行才发现。已经踩过两次，所以做成硬门禁：

    python tools/make_notebook.py && python tools/check_notebook.py

退出码非 0 表示有问题。
"""
import ast
import json
import pathlib
import subprocess
import sys
import tempfile

NB = pathlib.Path(__file__).resolve().parent.parent / "colab.ipynb"


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    bad = sh = py = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])

        if src.startswith("%%"):
            # 单元格魔法：交给 bash -n
            sh += 1
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
                fh.write("\n".join(src.splitlines()[1:]))
                path = fh.name
            r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
            pathlib.Path(path).unlink()
            if r.returncode:
                print(f"格 {i} bash 语法错误: {r.stderr.strip()[:200]}")
                bad += 1
        else:
            py += 1
            lines = [l for l in src.splitlines() if not l.strip().startswith("!")]
            try:
                ast.parse("\n".join(lines))
            except SyntaxError as e:
                print(f"格 {i} python 语法错误: {e}")
                bad += 1

    print(f"bash 格 {sh} / python 格 {py} / 问题 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
