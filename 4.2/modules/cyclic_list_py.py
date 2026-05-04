"""
Модуль: cyclic_list_py.py
Реализация циклического односвязного списка на чистом Python.
Лабораторная работа 4.2
"""


class _Node:
    """Узел циклического списка."""
    __slots__ = ("data", "next")

    def __init__(self, data: int):
        self.data: int = data
        self.next: "_Node | None" = None


class CyclicListPy:
    """
    Циклический односвязный список (Python-реализация).
    Последний узел хранит ссылку на head.
    """

    def __init__(self):
        self._head: _Node | None = None
        self._size: int = 0

    # ── Добавление в конец ────────────────────────────────────────────────────
    def add(self, value: int) -> None:
        new_node = _Node(value)
        if self._head is None:
            new_node.next = new_node          # единственный узел → сам на себя
            self._head = new_node
        else:
            current = self._head
            while current.next is not self._head:
                current = current.next
            current.next = new_node
            new_node.next = self._head
        self._size += 1

    # ── Удаление первого вхождения value ─────────────────────────────────────
    def remove(self, value: int) -> bool:
        if self._head is None:
            return False

        prev = None
        current = self._head

        while True:
            if current.data == value:
                if prev is None:                      # удаляем head
                    if current.next is current:       # единственный узел
                        self._head = None
                    else:
                        last = self._head
                        while last.next is not self._head:
                            last = last.next
                        last.next = self._head.next
                        self._head = self._head.next
                else:
                    prev.next = current.next
                self._size -= 1
                return True

            prev = current
            current = current.next
            if current is self._head:
                break

        return False

    # ── Вспомогательные методы ────────────────────────────────────────────────
    def get_elements(self) -> list[int]:
        """Вернуть все элементы в порядке обхода."""
        if self._head is None:
            return []
        result: list[int] = []
        current = self._head
        while True:
            result.append(current.data)
            current = current.next
            if current is self._head:
                break
        return result

    def get_size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._head is None

    def clear(self) -> None:
        self._head = None
        self._size = 0
