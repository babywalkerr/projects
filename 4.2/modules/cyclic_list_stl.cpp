/*
 * cyclic_list_stl.cpp
 * Циклический список — реализация на контейнерах STL (std::list).
 * Логика цикличности эмулируется: «последний→первый» реализован через
 * итераторы std::list и wrap-around доступ.
 *
 * Компиляция (Windows/MinGW):
 *   g++ -shared -o cyclic_list_stl.dll cyclic_list_stl.cpp -std=c++17
 * Компиляция (Linux):
 *   g++ -shared -fPIC -o cyclic_list_stl.so cyclic_list_stl.cpp -std=c++17
 */

#include <list>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

// Используем std::list<int> как внутреннее хранилище.
// Цикличность (last->next == head) — семантическая: при обходе wrap-around
// выполняется автоматически через итераторы с проверкой конца списка.

using Container = std::list<int>;

EXPORT void* create_list() {
    return new Container();
}

EXPORT void destroy_list(void* handle) {
    delete reinterpret_cast<Container*>(handle);
}

EXPORT void add_element(void* handle, int value) {
    reinterpret_cast<Container*>(handle)->push_back(value);
}

// Возвращает 1 если найден и удалён, 0 иначе.
EXPORT int remove_element(void* handle, int value) {
    auto* lst = reinterpret_cast<Container*>(handle);
    auto  it  = std::find(lst->begin(), lst->end(), value);
    if (it != lst->end()) {
        lst->erase(it);
        return 1;
    }
    return 0;
}

EXPORT int get_size(void* handle) {
    return static_cast<int>(reinterpret_cast<Container*>(handle)->size());
}

EXPORT void get_elements(void* handle, int* buffer, int max_size) {
    auto* lst = reinterpret_cast<Container*>(handle);
    int   i   = 0;
    for (int val : *lst) {
        if (i >= max_size) break;
        buffer[i++] = val;
    }
}

EXPORT void clear_list(void* handle) {
    reinterpret_cast<Container*>(handle)->clear();
}
