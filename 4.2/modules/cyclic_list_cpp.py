"""
Модуль: cyclic_list_cpp.py
ctypes-адаптер для C++ разделяемых библиотек.
Предоставляет единый интерфейс независимо от того, используется
dynamic-structs DLL или STL DLL.

Интерфейс идентичен CyclicListPy:
  add(value)       → None
  remove(value)    → bool
  get_elements()   → list[int]
  get_size()       → int
  is_empty()       → bool
  clear()          → None
"""

import ctypes
import os
import platform


def _dll_name(base: str) -> str:
    """Вернуть платформо-зависимое имя библиотеки."""
    system = platform.system()
    if system == "Windows":
        return f"{base}.dll"
    elif system == "Darwin":
        return f"{base}.dylib"   # macOS: без префикса lib (как собирает build.sh)
    else:
        return f"lib{base}.so"


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Объявить типы аргументов и возвращаемых значений функций DLL."""
    lib.create_list.restype  = ctypes.c_void_p
    lib.create_list.argtypes = []

    lib.destroy_list.restype  = None
    lib.destroy_list.argtypes = [ctypes.c_void_p]

    lib.add_element.restype  = None
    lib.add_element.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.remove_element.restype  = ctypes.c_int
    lib.remove_element.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.get_size.restype  = ctypes.c_int
    lib.get_size.argtypes = [ctypes.c_void_p]

    lib.get_elements.restype  = None
    lib.get_elements.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

    lib.clear_list.restype  = None
    lib.clear_list.argtypes = [ctypes.c_void_p]


class CyclicListCpp:
    """
    Обёртка над C++ DLL.
    Параметр dll_name: 'cyclic_list' (dynamic) или 'cyclic_list_stl' (STL).
    """

    def __init__(self, dll_name: str):
        # Ищем DLL в директории этого модуля
        module_dir = os.path.dirname(os.path.abspath(__file__))
        dll_path   = os.path.join(module_dir, _dll_name(dll_name))

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"Библиотека не найдена: {dll_path}")

        self._lib    = ctypes.CDLL(dll_path)
        _setup_signatures(self._lib)
        self._handle = self._lib.create_list()

    # ── Операции ─────────────────────────────────────────────────────────────

    def add(self, value: int) -> None:
        self._lib.add_element(self._handle, value)

    def remove(self, value: int) -> bool:
        return bool(self._lib.remove_element(self._handle, value))

    def get_elements(self) -> list[int]:
        size = self._lib.get_size(self._handle)
        if size == 0:
            return []
        buf = (ctypes.c_int * size)()
        self._lib.get_elements(self._handle, buf, size)
        return list(buf)

    def get_size(self) -> int:
        return self._lib.get_size(self._handle)

    def is_empty(self) -> bool:
        return self._lib.get_size(self._handle) == 0

    def clear(self) -> None:
        self._lib.clear_list(self._handle)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_handle") and self._handle:
            self._lib.destroy_list(self._handle)
            self._handle = None
