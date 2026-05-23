"""
Лабораторная работа №7. Создание интерактивного дашборда.
Вариант 16: Лазерные измерения.

Стек: tkinter + matplotlib + seaborn + pandas.
Архитектура: процедурная (без классов), событийно-ориентированная.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ════════════════════════════════════════════════════════════════
# Конфигурация
# ════════════════════════════════════════════════════════════════
VARIANT_NUMBER = 16
DATA_PATH = "data_dashboard.csv"

# Кириллица в matplotlib
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# ════════════════════════════════════════════════════════════════
# Глобальное состояние
# ════════════════════════════════════════════════════════════════
df_raw = None             # исходные данные
df_work = None            # рабочая копия после фильтрации
fig = plt.Figure(figsize=(9, 5.5), dpi=100)
canvas = None
current_chart = "line"    # текущий тип графика

# Контролы (переменные tk)
var_laser = None
var_shift = None
var_agg = None
var_smoothing = None
var_resample = None


# ════════════════════════════════════════════════════════════════
# Этап 2. Предобработка и Feature Engineering
# ════════════════════════════════════════════════════════════════
def preprocess_data():
    """Изолированная обработка: фильтрация, новые признаки, IQR-чистка."""
    global df_raw

    work = df_raw.copy()

    # 1. Фильтрация по условию варианта 16: err == 0 (уже сделано в generate)
    work = work[work["err"] == 0].copy()

    # 2. Безопасное вычисление производного признака
    # Энергия импульса = power / pulse, защита от деления на ноль
    denom = np.where(work["pulse"].values > 0, work["pulse"].values, 1e-8)
    work["energy_per_pulse"] = work["power"].values / denom
    # Заменяем inf / NaN на 0
    work["energy_per_pulse"] = (
        work["energy_per_pulse"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # 3. Обрезка выбросов IQR по группам (laser_type)
    def iqr_clip(group, col):
        q1, q3 = group[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return group[col].clip(low, high)

    for col in ("power", "pulse", "wave"):
        work[col] = (
            work.groupby("laser_type", group_keys=False)
            .apply(lambda g: iqr_clip(g, col))
        )

    # 4. Оптимизация категориальных полей (экономит память и ускоряет фильтры)
    for cat_col in ("shift", "laser_type", "status"):
        work[cat_col] = work[cat_col].astype("category")

    # Дату делаем индексом для resample/rolling
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)

    return work


# ════════════════════════════════════════════════════════════════
# Применение пользовательских фильтров (запускается перед каждой отрисовкой)
# ════════════════════════════════════════════════════════════════
def apply_user_filters(df):
    """Возвращает срез данных с учётом текущих значений виджетов."""
    out = df.copy()

    # Фильтр по типу лазера (Combobox)
    laser = var_laser.get()
    if laser != "Все":
        out = out.query("laser_type == @laser")

    # Фильтр по смене (Combobox)
    shift = var_shift.get()
    if shift != "Все":
        out = out.query("shift == @shift")

    return out.reset_index(drop=True) if len(out) else out


# ════════════════════════════════════════════════════════════════
# Этап 4. Функции отрисовки графиков (4 типа + дополнительный)
# ════════════════════════════════════════════════════════════════
def clear_figure():
    """Полная очистка фигуры перед новой отрисовкой."""
    fig.clear()


def plot_line():
    """Линейный график: средняя мощность по времени с настраиваемой агрегацией."""
    clear_figure()
    data = apply_user_filters(df_work)
    if len(data) == 0:
        _draw_empty("Нет данных для выбранных фильтров")
        return

    rule = var_resample.get()  # "D" | "W" | "ME"
    agg_func = var_agg.get()   # "mean" | "sum" | "median"

    # Этап 6: resample по периодам
    grouped = (
        data.set_index("date")["power"]
        .resample(rule)
        .agg(agg_func)
        .dropna()
    )

    # Скользящее сглаживание (если включено)
    if var_smoothing.get():
        window = max(3, len(grouped) // 20)
        grouped_smooth = grouped.rolling(window=window, min_periods=1).mean()
    else:
        grouped_smooth = grouped

    ax = fig.add_subplot(111)
    sns.lineplot(
        x=grouped_smooth.index, y=grouped_smooth.values,
        ax=ax, linewidth=2, color="#1f77b4",
    )
    ax.set_title(
        f"Динамика мощности (агрегация: {agg_func}, шаг: {rule})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Дата")
    ax.set_ylabel("Мощность, Вт")
    fig.autofmt_xdate()
    fig.tight_layout()
    canvas.draw_idle()


def plot_bar():
    """Столбчатая диаграмма: агрегация power по типам лазеров и сменам."""
    clear_figure()
    data = apply_user_filters(df_work)
    if len(data) == 0:
        _draw_empty("Нет данных для выбранных фильтров")
        return

    agg_func = var_agg.get()

    # Этап 6: pivot_table для подготовки матрицы
    pivot = data.pivot_table(
        values="power", index="laser_type", columns="shift",
        aggfunc=agg_func, observed=True,
    ).fillna(0)

    # melt - чтобы передать в seaborn в "длинном" формате
    long_df = pivot.reset_index().melt(
        id_vars="laser_type", var_name="shift", value_name="power",
    )

    ax = fig.add_subplot(111)
    sns.barplot(
        data=long_df, x="laser_type", y="power", hue="shift",
        ax=ax, palette="viridis",
    )
    ax.set_title(
        f"Мощность по типам лазеров и сменам ({agg_func})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Тип лазера")
    ax.set_ylabel(f"Мощность ({agg_func})")
    ax.legend(title="Смена", loc="upper right")
    fig.tight_layout()
    canvas.draw_idle()


def plot_scatter():
    """Точечная диаграмма: power vs pulse, цвет — статус."""
    clear_figure()
    data = apply_user_filters(df_work)
    if len(data) == 0:
        _draw_empty("Нет данных для выбранных фильтров")
        return

    # Подвыборка чтобы не задушить рендер
    if len(data) > 3000:
        data = data.sample(n=3000, random_state=42)

    ax = fig.add_subplot(111)
    sns.scatterplot(
        data=data, x="pulse", y="power", hue="status",
        ax=ax, palette={"OK": "#2ca02c", "WARNING": "#ff7f0e", "ERROR": "#d62728"},
        alpha=0.6, s=25,
    )
    ax.set_title(
        "Мощность vs частота импульсов (цвет — статус)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Частота импульсов, Гц")
    ax.set_ylabel("Мощность, Вт")
    ax.legend(title="Статус", loc="upper right")
    fig.tight_layout()
    canvas.draw_idle()


def plot_heatmap():
    """Тепловая карта: средняя мощность по сменам и типам лазеров."""
    clear_figure()
    data = apply_user_filters(df_work)
    if len(data) == 0:
        _draw_empty("Нет данных для выбранных фильтров")
        return

    agg_func = var_agg.get()

    # Этап 6: pivot_table для матрицы тепловой карты
    pivot = data.pivot_table(
        values="power", index="laser_type", columns="shift",
        aggfunc=agg_func, observed=True,
    ).fillna(0)

    ax = fig.add_subplot(111)
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="coolwarm",
        cbar_kws={"label": f"Мощность ({agg_func})"},
        ax=ax,
    )
    ax.set_title(
        f"Тепловая карта: тип лазера × смена ({agg_func})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Смена")
    ax.set_ylabel("Тип лазера")
    fig.tight_layout()
    canvas.draw_idle()


def plot_hist():
    """Гистограмма распределения мощности с биннингом."""
    clear_figure()
    data = apply_user_filters(df_work)
    if len(data) == 0:
        _draw_empty("Нет данных для выбранных фильтров")
        return

    # Этап 6: pd.cut - биннинг непрерывной величины на интервалы
    bins = pd.cut(data["power"], bins=15)
    counts = bins.value_counts().sort_index()

    ax = fig.add_subplot(111)
    sns.histplot(
        data=data, x="power", bins=15, hue="laser_type",
        ax=ax, palette="Set2", multiple="stack", edgecolor="white",
    )
    ax.set_title(
        "Распределение мощности по типам лазеров",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Мощность, Вт")
    ax.set_ylabel("Количество замеров")
    fig.tight_layout()
    canvas.draw_idle()


def _draw_empty(msg):
    """Пустой график-заглушка с сообщением."""
    ax = fig.add_subplot(111)
    ax.text(
        0.5, 0.5, msg, ha="center", va="center",
        fontsize=14, color="#888888", transform=ax.transAxes,
    )
    ax.set_xticks([]); ax.set_yticks([])
    canvas.draw_idle()


# ════════════════════════════════════════════════════════════════
# Этап 5. Обработчики панели управления
# ════════════════════════════════════════════════════════════════
def refresh_data():
    """Полный пересчёт данных и перерисовка текущего графика."""
    global df_work
    df_work = preprocess_data()
    redraw_current()


def redraw_current(*_):
    """Перерисовка текущего графика — вызывается из callback'ов виджетов."""
    chart_funcs = {
        "line": plot_line,
        "bar": plot_bar,
        "scatter": plot_scatter,
        "heatmap": plot_heatmap,
        "hist": plot_hist,
    }
    func = chart_funcs.get(current_chart, plot_line)
    func()


def set_chart(chart_name):
    """Переключение типа активного графика."""
    global current_chart
    current_chart = chart_name
    redraw_current()


def export_plot():
    """Экспорт текущей фигуры в PNG/PDF через filedialog."""
    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
        initialfile=f"dashboard_{current_chart}.png",
    )
    if filepath:
        try:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Экспорт", f"Сохранено: {filepath}")
        except Exception as e:
            messagebox.showerror("Ошибка экспорта", str(e))


# ════════════════════════════════════════════════════════════════
# Этап 1 + 3. Загрузка данных и сборка интерфейса
# ════════════════════════════════════════════════════════════════
def main():
    global df_raw, df_work, canvas
    global var_laser, var_shift, var_agg, var_smoothing, var_resample

    # ── загрузка ────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    except FileNotFoundError:
        print(f"Ошибка: не найден {DATA_PATH}")
        print("Сначала запусти: python generate_dashboard_data.py")
        return

    print(f"Загружено: {len(df_raw):,} строк")

    # ── главное окно ────────────────────────────────────────
    root = tk.Tk()
    root.title(f"Дашборд: Вариант {VARIANT_NUMBER} — Лазерные измерения")
    root.geometry("1280x800")
    root.configure(bg="#f0f2f5")

    # ── переменные виджетов ────────────────────────────────
    var_laser = tk.StringVar(value="Все")
    var_shift = tk.StringVar(value="Все")
    var_agg = tk.StringVar(value="mean")
    var_smoothing = tk.BooleanVar(value=False)
    var_resample = tk.StringVar(value="W")

    # ── верхняя панель фильтров ────────────────────────────
    filter_frame = tk.Frame(root, bg="#dde8f5", pady=8)
    filter_frame.pack(fill=tk.X)

    tk.Label(
        filter_frame, text="Тип лазера:", bg="#dde8f5",
        font=("Helvetica", 10, "bold"),
    ).pack(side=tk.LEFT, padx=(12, 4))
    laser_combo = ttk.Combobox(
        filter_frame, textvariable=var_laser, width=12, state="readonly",
        values=["Все"] + sorted(df_raw["laser_type"].unique().tolist()),
    )
    laser_combo.pack(side=tk.LEFT, padx=4)
    laser_combo.bind("<<ComboboxSelected>>", redraw_current)

    tk.Label(
        filter_frame, text="Смена:", bg="#dde8f5",
        font=("Helvetica", 10, "bold"),
    ).pack(side=tk.LEFT, padx=(16, 4))
    shift_combo = ttk.Combobox(
        filter_frame, textvariable=var_shift, width=12, state="readonly",
        values=["Все"] + sorted(df_raw["shift"].unique().tolist()),
    )
    shift_combo.pack(side=tk.LEFT, padx=4)
    shift_combo.bind("<<ComboboxSelected>>", redraw_current)

    # RadioButtons для агрегации
    tk.Label(
        filter_frame, text="Агрегация:", bg="#dde8f5",
        font=("Helvetica", 10, "bold"),
    ).pack(side=tk.LEFT, padx=(16, 4))
    for label, value in [("Mean", "mean"), ("Sum", "sum"), ("Median", "median")]:
        tk.Radiobutton(
            filter_frame, text=label, variable=var_agg, value=value,
            bg="#dde8f5", command=redraw_current,
        ).pack(side=tk.LEFT, padx=2)

    # Combobox для шага resample
    tk.Label(
        filter_frame, text="Шаг:", bg="#dde8f5",
        font=("Helvetica", 10, "bold"),
    ).pack(side=tk.LEFT, padx=(16, 4))
    resample_combo = ttk.Combobox(
        filter_frame, textvariable=var_resample, width=10, state="readonly",
        values=[("D"), ("W"), ("ME"), ("YE")],
    )
    resample_combo.pack(side=tk.LEFT, padx=4)
    resample_combo.bind("<<ComboboxSelected>>", redraw_current)

    # Чекбокс сглаживания
    tk.Checkbutton(
        filter_frame, text="Сглаживание (rolling)",
        variable=var_smoothing, bg="#dde8f5",
        command=redraw_current,
    ).pack(side=tk.LEFT, padx=(16, 4))

    # ── центральный фрейм для графика ──────────────────────
    plot_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=1)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    # Адаптер matplotlib → tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)

    # ── нижняя панель кнопок-переключателей графиков ───────
    ctrl_frame = tk.Frame(root, bg="#f0f2f5", pady=8)
    ctrl_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

    chart_buttons = [
        ("Линейный", "line", "#4caf50"),
        ("Столбчатый", "bar", "#1e88e5"),
        ("Точечный", "scatter", "#ab47bc"),
        ("Тепловая карта", "heatmap", "#fb8c00"),
        ("Гистограмма", "hist", "#26a69a"),
    ]
    for label, name, color in chart_buttons:
        tk.Button(
            ctrl_frame, text=label, width=14,
            bg=color, fg="white", font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=lambda n=name: set_chart(n),
        ).pack(side=tk.LEFT, padx=4)

    # справа — служебные кнопки
    tk.Button(
        ctrl_frame, text="Обновить", width=12,
        bg="#607d8b", fg="white", font=("Helvetica", 10, "bold"),
        relief=tk.FLAT, cursor="hand2", command=refresh_data,
    ).pack(side=tk.RIGHT, padx=4)

    tk.Button(
        ctrl_frame, text="Экспорт", width=12,
        bg="#f4511e", fg="white", font=("Helvetica", 10, "bold"),
        relief=tk.FLAT, cursor="hand2", command=export_plot,
    ).pack(side=tk.RIGHT, padx=4)

    # ── первичная отрисовка ────────────────────────────────
    df_work = preprocess_data()
    plot_line()

    root.mainloop()


if __name__ == "__main__":
    main()
