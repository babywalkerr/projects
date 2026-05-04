#!/usr/bin/env bash
# build.sh — Компиляция C++ модулей (Linux / macOS)
# Требует: g++

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " Lab 4.2 — Сборка C++ модулей"
echo "============================================================"
echo ""

# Определяем расширение библиотеки
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    EXT="dylib"
    FLAGS="-dynamiclib -undefined dynamic_lookup"
else
    EXT="so"
    FLAGS="-shared -fPIC"
fi

# ── Модуль 1: Dynamic structs ─────────────────────────────────────────────
echo "[1/2] Компиляция cyclic_list.$EXT (dynamic structs)..."
g++ $FLAGS -o "modules/cyclic_list.$EXT" modules/cyclic_list.cpp -std=c++17
echo "  OK: modules/cyclic_list.$EXT"

# ── Модуль 2: STL ─────────────────────────────────────────────────────────
echo ""
echo "[2/2] Компиляция cyclic_list_stl.$EXT (STL)..."
g++ $FLAGS -o "modules/cyclic_list_stl.$EXT" modules/cyclic_list_stl.cpp -std=c++17
echo "  OK: modules/cyclic_list_stl.$EXT"

echo ""
echo "============================================================"
echo " Сборка завершена. Запустите: python3 main.py"
echo "============================================================"
