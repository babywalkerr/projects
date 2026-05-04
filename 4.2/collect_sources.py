"""
collect_sources.py — собирает все исходники проекта в один файл.
Запуск: python3 collect_sources.py
Результат: all_sources.txt
"""

import os

FILES = [
    "main.py",
    "modules/cyclic_list_py.py",
    "modules/cyclic_list_cpp.py",
    "modules/cyclic_list.cpp",
    "modules/cyclic_list_stl.cpp",
]

root = os.path.dirname(os.path.abspath(__file__))
out  = os.path.join(root, "all_sources.txt")

with open(out, "w", encoding="utf-8") as f:
    for path in FILES:
        full = os.path.join(root, path)
        f.write(f"{'='*60}\n")
        f.write(f"FILE: {path}\n")
        f.write(f"{'='*60}\n\n")
        with open(full, encoding="utf-8") as src:
            f.write(src.read())
        f.write("\n\n")

print(f"Готово: {out}")
