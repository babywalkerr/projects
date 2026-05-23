"""
Генератор расширенного датасета для лабораторной 7 (дашборд).
На вход: data_variant_16.csv (исходные данные лабы 6).
На выход: data_dashboard.csv с 4 дополнительными категориальными признаками:

    1. date       (datetime) - дата замера, выведена из ts
    2. shift      (text)     - смена: morning / day / night, по часу замера
    3. laser_type (text)     - тип лазера, привязан к beam_id (детерминированно)
    4. status     (text)     - статус замера: OK / WARNING / ERROR, по err

Дополнительно делаем выборку 50 000 строк, чтобы дашборд работал быстро.
Применяем фильтр err == 0 заранее (акцент варианта 16) - так данные в дашборде
сразу пригодны для аналитики.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SRC = "data_variant_16.csv"
DST = "data_dashboard.csv"
SAMPLE_SIZE = 50_000
RNG_SEED = 16

LASER_TYPES = ["CO2", "Nd:YAG", "Fiber", "Diode", "Excimer"]


def main() -> None:
    print(f"Чтение исходного CSV: {SRC}")
    # bad_lines=skip - в исходном CSV есть битые строки
    df = pd.read_csv(SRC, on_bad_lines="skip")
    print(f"  Прочитано: {len(df):,} строк")

    # Очистка: убираем NaN, оставляем только err == 0 (акцент варианта)
    df = df.dropna()
    df = df[df["err"] == 0].reset_index(drop=True)
    print(f"  После dropna и err==0: {len(df):,} строк")

    # Базовая очистка по диапазонам (как в лабе 6)
    df = df[df["power"] >= 0]
    df = df[df["pulse"] >= 0]
    df = df[(df["wave"] > 0) & (df["wave"] < 1000)]
    df = df.reset_index(drop=True)
    print(f"  После базовой очистки: {len(df):,} строк")

    # Подвыборка
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RNG_SEED).reset_index(drop=True)
        print(f"  Подвыборка {SAMPLE_SIZE:,} строк")

    rng = np.random.default_rng(RNG_SEED)

    # ── 1. date - из ts (UNIX timestamp) ────────────────────
    # ts хранится как int32 в секундах от эпохи. Превращаем в datetime.
    df["date"] = pd.to_datetime(df["ts"], unit="s")

    # ── 2. shift - утро / день / ночь по часу замера ────────
    hours = df["date"].dt.hour
    shift = pd.Series("night", index=df.index, dtype="string")
    shift[(hours >= 6) & (hours < 14)] = "morning"
    shift[(hours >= 14) & (hours < 22)] = "day"
    df["shift"] = shift

    # ── 3. laser_type - детерминированно по beam_id ─────────
    # Чтобы при повторных запусках было одно и то же.
    df["laser_type"] = df["beam_id"].apply(
        lambda b: LASER_TYPES[int(b) % len(LASER_TYPES)]
    )

    # ── 4. status - по комбинации err и power ───────────────
    # err == 0 у нас всегда, но добавим лёгкий шум через power-аномалии
    p_low, p_high = df["power"].quantile([0.05, 0.95])
    status = pd.Series("OK", index=df.index, dtype="string")
    status[(df["power"] < p_low) | (df["power"] > p_high)] = "WARNING"
    # Случайный 1% переведём в ERROR — для разнообразия категорий
    err_idx = rng.choice(df.index, size=max(1, len(df) // 100), replace=False)
    status.loc[err_idx] = "ERROR"
    df["status"] = status

    # Сортируем по дате - важно для time-series графиков
    df = df.sort_values("date").reset_index(drop=True)

    # Сохраняем
    df.to_csv(DST, index=False)
    print(f"\nСохранено: {DST}")
    print(f"  Размер: {Path(DST).stat().st_size / 1024:.1f} КБ")
    print(f"  Строк:  {len(df):,}")
    print(f"  Поля:   {list(df.columns)}")
    print()
    print("Распределение категорий:")
    print(f"  shift:      {df['shift'].value_counts().to_dict()}")
    print(f"  laser_type: {df['laser_type'].value_counts().to_dict()}")
    print(f"  status:     {df['status'].value_counts().to_dict()}")
    print(f"  date range: {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    main()
