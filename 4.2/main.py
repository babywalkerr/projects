"""
main.py — GUI для циклического односвязного списка
Лабораторная работа 4.2

Запуск: python3 main.py
"""

import tkinter as tk
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.cyclic_list_py import CyclicListPy

try:
    from modules.cyclic_list_cpp import CyclicListCpp
    _CPP_OK = True
except Exception:
    _CPP_OK = False

MODULE_LABELS = {
    "python":      "Python",
    "cpp_dynamic": "C++ (dynamic structs)",
    "cpp_stl":     "C++ (STL std::list)",
}

# Цвета — все явные, не зависят от системной темы
C_BG        = "#f5f5f5"
C_PANEL     = "#ffffff"
C_HEADER    = "#2c6fad"
C_HEADER2   = "#dde8f5"
C_TEXT      = "#1a1a1a"
C_GRAY      = "#555555"
C_BLUE      = "#1e6fba"
C_GREEN_BTN = "#4caf50"
C_RED_BTN   = "#e53935"
C_ORG_BTN   = "#fb8c00"
C_WHITE     = "#ffffff"
C_BORDER    = "#b0b8c4"


def _load_module(key):
    if key == "python":
        return CyclicListPy()
    if not _CPP_OK:
        return None
    dll_map = {"cpp_dynamic": "cyclic_list", "cpp_stl": "cyclic_list_stl"}
    try:
        return CyclicListCpp(dll_map[key])
    except Exception:
        return None


def _btn(parent, text, bg, command):
    """Кнопка в цветной рамке — работает корректно на macOS."""
    border = tk.Frame(parent, bg=bg, padx=2, pady=2)
    tk.Button(
        border, text=text, command=command,
        bg=bg, fg="#1a1a1a", relief=tk.FLAT,
        font=("Helvetica", 11, "bold"), pady=5,
        activebackground=bg, activeforeground="#1a1a1a",
        cursor="hand2", bd=0,
    ).pack(fill=tk.X)
    return border


def _label(parent, text, font=None, fg=None, anchor=tk.W):
    return tk.Label(
        parent, text=text, bg=C_PANEL,
        fg=fg or C_TEXT,
        font=font or ("Helvetica", 10),
        anchor=anchor,
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа 4.2 — Циклический список")
        self.configure(bg=C_BG)
        self.minsize(860, 520)
        self.geometry("1050x630")

        self._modules = {}
        for key in ("python", "cpp_dynamic", "cpp_stl"):
            m = _load_module(key)
            if m is not None:
                self._modules[key] = m

        self._module_key = tk.StringVar(value="python")

        self._build_ui()
        self.update()
        self._refresh()
        self.canvas.bind("<Configure>", lambda _: self._refresh())

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_module_bar()

        main = tk.Frame(self, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self._build_controls(main)
        self._build_canvas_area(main)

        self._status_var = tk.StringVar(value="Готово")
        tk.Label(
            self, textvariable=self._status_var,
            bg="#e0e0e0", fg=C_TEXT, anchor=tk.W,
            font=("Helvetica", 9), pady=4, padx=10,
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_header(self):
        f = tk.Frame(self, bg=C_HEADER, pady=10)
        f.pack(fill=tk.X)
        tk.Label(f, text="Циклический односвязный список",
                 bg=C_HEADER, fg=C_WHITE,
                 font=("Helvetica", 14, "bold")).pack(side=tk.LEFT, padx=14)
        tk.Label(f, text="Лабораторная работа 4.2",
                 bg=C_HEADER, fg="#c8dfff",
                 font=("Helvetica", 10)).pack(side=tk.RIGHT, padx=14)

    def _build_module_bar(self):
        f = tk.Frame(self, bg=C_HEADER2, pady=7)
        f.pack(fill=tk.X)
        tk.Label(f, text="Модуль реализации:",
                 bg=C_HEADER2, fg=C_TEXT,
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=12)
        for key, label in MODULE_LABELS.items():
            available = key in self._modules
            text = label if available else f"{label}  [DLL не найдена]"
            tk.Radiobutton(
                f, text=text,
                variable=self._module_key, value=key,
                state=tk.NORMAL if available else tk.DISABLED,
                bg=C_HEADER2, fg=C_TEXT,
                selectcolor=C_HEADER2,
                activebackground=C_HEADER2,
                font=("Helvetica", 10),
                command=self._on_module_change,
            ).pack(side=tk.LEFT, padx=10)

    def _build_controls(self, parent):
        panel = tk.Frame(parent, bg=C_PANEL, width=240,
                         highlightbackground=C_BORDER,
                         highlightthickness=1)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        panel.pack_propagate(False)

        # ── Операции ─────────────────────────────────────────────────────────
        self._section(panel, "Операции")

        tk.Label(panel, text="Значение:", bg=C_PANEL, fg=C_GRAY,
                 font=("Helvetica", 9)).pack(anchor=tk.W, padx=12, pady=(4, 2))

        self._entry = tk.Entry(
            panel, font=("Helvetica", 12),
            bg=C_WHITE, fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.SOLID, bd=1,
        )
        self._entry.pack(fill=tk.X, padx=12, pady=(0, 8))
        self._entry.bind("<Return>", lambda _: self._op_add())

        _btn(panel, "Добавить",       C_GREEN_BTN, self._op_add   ).pack(fill=tk.X, padx=12, pady=2)
        _btn(panel, "Удалить",        C_RED_BTN,   self._op_remove).pack(fill=tk.X, padx=12, pady=2)
        tk.Frame(panel, bg=C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=8)
        _btn(panel, "Очистить список", C_ORG_BTN,  self._op_clear ).pack(fill=tk.X, padx=12, pady=2)

        tk.Frame(panel, bg=C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=10)

        # ── Информация ───────────────────────────────────────────────────────
        self._section(panel, "Информация")

        def _info_row(label, val_color):
            r = tk.Frame(panel, bg=C_PANEL)
            r.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(r, text=label, bg=C_PANEL, fg=C_GRAY,
                     font=("Helvetica", 9)).pack(side=tk.LEFT)
            lbl = tk.Label(r, text="—", bg=C_PANEL, fg=val_color,
                           font=("Helvetica", 9, "bold"))
            lbl.pack(side=tk.LEFT, padx=6)
            return lbl

        self._lbl_size   = _info_row("Размер:", C_BLUE)
        self._lbl_module = _info_row("Модуль:", C_BLUE)

        tk.Frame(panel, bg=C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=10)

        # ── Легенда ──────────────────────────────────────────────────────────
        self._section(panel, "Обозначения")

        for color, text in [
            (C_BLUE,    "■   узел списка"),
            ("#2e7d32", "→   стрелка next"),
            ("#b8860b", "- →  last→head (цикл)"),
        ]:
            tk.Label(panel, text=text, bg=C_PANEL, fg=color,
                     font=("Helvetica", 9)).pack(anchor=tk.W, padx=14, pady=2)

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=C_PANEL, fg=C_BLUE,
                 font=("Helvetica", 10, "bold")).pack(
            anchor=tk.W, padx=12, pady=(8, 2))

    def _build_canvas_area(self, parent):
        right = tk.Frame(parent, bg=C_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Визуализация", bg=C_BG, fg=C_TEXT,
                 font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))

        self.canvas = tk.Canvas(
            right, bg=C_WHITE,
            highlightbackground=C_BORDER,
            highlightthickness=1,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ── Логика ────────────────────────────────────────────────────────────────

    def _list(self):
        return self._modules[self._module_key.get()]

    def _on_module_change(self):
        self._set_status(f"Модуль: {MODULE_LABELS[self._module_key.get()]}")
        self._refresh()

    def _op_add(self):
        raw = self._entry.get().strip()
        if not raw:
            self._set_status("Ошибка: введите целое число"); return
        try:
            n = int(raw)
        except ValueError:
            self._set_status("Ошибка: ожидается целое число"); return
        self._list().add(n)
        self._entry.delete(0, tk.END)
        self._set_status(f"Элемент {n} добавлен")
        self._refresh()

    def _op_remove(self):
        raw = self._entry.get().strip()
        if not raw:
            self._set_status("Ошибка: введите значение для удаления"); return
        try:
            n = int(raw)
        except ValueError:
            self._set_status("Ошибка: ожидается целое число"); return
        if self._list().is_empty():
            self._set_status("Ошибка: список пуст — нечего удалять"); return
        if self._list().remove(n):
            self._entry.delete(0, tk.END)
            self._set_status(f"Элемент {n} удалён")
        else:
            self._set_status(f"Ошибка: элемент {n} не найден в списке")
        self._refresh()

    def _op_clear(self):
        if self._list().is_empty():
            self._set_status("Список уже пуст"); return
        self._list().clear()
        self._set_status("Список очищен")
        self._refresh()

    def _set_status(self, msg):
        self._status_var.set(msg)

    def _refresh(self):
        lst = self._list()
        elements = lst.get_elements()
        self._lbl_size.configure(text=str(len(elements)))
        self._lbl_module.configure(
            text=MODULE_LABELS.get(self._module_key.get(), ""))
        self._draw(elements)

    # ── Визуализация ──────────────────────────────────────────────────────────

    def _draw(self, elements):
        self.update_idletasks()
        c = self.canvas
        c.delete("all")

        W = c.winfo_width()
        H = c.winfo_height()
        if W <= 1 or H <= 1:
            self.after(50, lambda: self._draw(elements))
            return

        n = len(elements)

        if n == 0:
            c.create_text(W / 2, H / 2 - 14, text="Список пуст",
                          fill="#aaaaaa", font=("Helvetica", 15, "italic"))
            c.create_text(W / 2, H / 2 + 14,
                          text="Введите число и нажмите «Добавить»",
                          fill="#bbbbbb", font=("Helvetica", 10))
            return

        node_r = max(22, min(36, int(min(W, H) * 0.36 / max(n, 1))))
        R      = max(node_r * 2.8, min(W, H) / 2 - node_r * 2.6)
        cx, cy = W / 2, H / 2

        def pos(i):
            a = math.pi / 2 - i * 2 * math.pi / n
            return cx + R * math.cos(a), cy - R * math.sin(a)

        positions = [pos(i) for i in range(n)]

        # Стрелки
        for i in range(n):
            x1, y1 = positions[i]
            x2, y2 = positions[(i + 1) % n]
            is_last = (i == n - 1)
            color = "#b8860b" if is_last else "#2e7d32"
            dash  = (6, 4)   if is_last else ()

            if n == 1:
                lx, ly = x1 + node_r + 6, y1 - node_r - 6
                r2 = node_r * 0.6
                c.create_oval(lx - r2, ly - r2, lx + r2, ly + r2,
                              outline=color, width=2, dash=(5, 3))
                c.create_line(lx, ly + r2,
                              x1 + node_r * 0.55, y1 - node_r * 0.55,
                              fill=color, width=2,
                              arrow=tk.LAST, arrowshape=(8, 10, 3))
                continue

            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            if dist < 1:
                continue
            ndx, ndy = dx / dist, dy / dist
            ax1 = x1 + ndx * (node_r + 4)
            ay1 = y1 + ndy * (node_r + 4)
            ax2 = x2 - ndx * (node_r + 10)
            ay2 = y2 - ndy * (node_r + 10)

            if is_last and n > 2:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                pcx = mx + (mx - cx) * 0.5
                pcy = my + (my - cy) * 0.5
                c.create_line(ax1, ay1, pcx, pcy, ax2, ay2,
                              smooth=True, fill=color, width=2,
                              arrow=tk.LAST, arrowshape=(9, 11, 4), dash=dash)
            else:
                c.create_line(ax1, ay1, ax2, ay2,
                              fill=color, width=2,
                              arrow=tk.LAST, arrowshape=(9, 11, 4), dash=dash)

        # Узлы
        for i, (x, y) in enumerate(positions):
            is_head = (i == 0)
            fill    = "#bbdefb" if is_head else "#e8f4fd"
            lw      = 3        if is_head else 2

            c.create_oval(x - node_r, y - node_r,
                          x + node_r, y + node_r,
                          fill=fill, outline=C_BLUE, width=lw)

            fsize = max(9, min(14, node_r - 6))
            c.create_text(x, y, text=str(elements[i]),
                          font=("Helvetica", fsize, "bold"), fill="#0d1b2a")

            if is_head:
                c.create_text(x, y - node_r - 12, text="head",
                              font=("Helvetica", 8, "bold"), fill=C_BLUE)

        c.create_text(W - 8, 8, text=f"Элементов: {n}",
                      anchor=tk.NE, fill="#999999", font=("Helvetica", 8))


if __name__ == "__main__":
    app = App()
    app.mainloop()
