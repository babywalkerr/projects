# Лабораторная работа 4.2 — GUI для циклического односвязного списка

> **Цель:** разработать оконное приложение на tkinter с визуализацией циклического односвязного списка, интегрировав Python-модуль и C++-модули через ctypes.

---

## Структура проекта

```
lab4_2/
├── main.py                     # Точка входа — tkinter GUI
├── build.bat                   # Сборка DLL (Windows / MinGW)
├── build.sh                    # Сборка .so/.dylib (Linux / macOS)
└── modules/
    ├── __init__.py
    ├── cyclic_list_py.py       # Python-реализация списка
    ├── cyclic_list_cpp.py      # ctypes-адаптер для обоих DLL
    ├── cyclic_list.cpp         # C++ — динамические структуры
    └── cyclic_list_stl.cpp     # C++ — STL (std::list)
```

---

## Быстрый старт

### 1. Запуск (только Python-модуль)
```bash
python main.py
```
Приложение работает **без компиляции C++** — Python-модуль доступен всегда.

### 2. Сборка C++ модулей (Windows, MinGW)
```bat
build.bat
```

### 2. Сборка C++ модулей (Linux / macOS)
```bash
chmod +x build.sh
./build.sh
```

### Требования
- Python 3.10+
- tkinter (входит в стандартную поставку CPython)
- g++ с поддержкой C++17 (только для C++ модулей)

---

## Модули

| Модуль | Файл | Описание |
|---|---|---|
| Python | `cyclic_list_py.py` | Чистый Python, ссылки на объекты `_Node` |
| C++ Dynamic | `cyclic_list.cpp` | Структуры `Node` + `CyclicList` на `new`/`delete` |
| C++ STL | `cyclic_list_stl.cpp` | `std::list<int>` с wrap-around семантикой |

Все три модуля предоставляют **одинаковый интерфейс**:

```python
list.add(value: int)        → None
list.remove(value: int)     → bool
list.get_elements()         → list[int]
list.get_size()             → int
list.is_empty()             → bool
list.clear()                → None
```

---

## Возможности GUI

- **Переключение модуля** — radiobutton-бар выбирает Python / C++ dynamic / C++ STL; каждый модуль хранит **независимые данные**
- **Добавление** — по значению или клавише Enter
- **Удаление** — по значению; корректное сообщение при пустом списке или отсутствии элемента
- **Очистка** — полный сброс списка
- **Визуализация** — узлы расположены по окружности:
  - `head`-узел выделен ореолом
  - Зелёные стрелки — направление `next`
  - Жёлтая пунктирная стрелка — цикл `last → head`
  - Адаптивный размер узлов под размер окна
- **Статусная строка** — сообщение об успехе / ошибке после каждой операции

---

## Архитектура

```
main.py (GUI / tkinter)
    │
    ├─── modules/cyclic_list_py.py        (Python-реализация)
    │
    └─── modules/cyclic_list_cpp.py       (ctypes-адаптер)
              │
              ├─── cyclic_list.dll/.so    (C++ dynamic structs)
              └─── cyclic_list_stl.dll/.so(C++ STL)
```

Вся работа со структурой данных выполняется **исключительно** через вызовы функций модулей — GUI не содержит логики списка.
