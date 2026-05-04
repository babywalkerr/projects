/*
 * cyclic_list.cpp
 * Циклический односвязный список — реализация на динамических структурах.
 * Экспортируется как разделяемая библиотека (DLL/so).
 *
 * Компиляция (Windows/MinGW):
 *   g++ -shared -o cyclic_list.dll cyclic_list.cpp -std=c++17
 * Компиляция (Linux):
 *   g++ -shared -fPIC -o cyclic_list.so cyclic_list.cpp -std=c++17
 */

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif

// ── Внутренние структуры ─────────────────────────────────────────────────────

struct Node {
    int   data;
    Node* next;
};

struct CyclicList {
    Node* head;
    int   size;
};

// ── API ───────────────────────────────────────────────────────────────────────

EXPORT void* create_list() {
    CyclicList* list = new CyclicList{nullptr, 0};
    return list;
}

EXPORT void destroy_list(void* handle) {
    if (!handle) return;
    CyclicList* list = reinterpret_cast<CyclicList*>(handle);

    if (list->head) {
        Node* current = list->head;
        Node* next_node;
        do {
            next_node = current->next;
            delete current;
            current = next_node;
        } while (current != list->head);
    }
    delete list;
}

EXPORT void add_element(void* handle, int value) {
    CyclicList* list = reinterpret_cast<CyclicList*>(handle);
    Node* new_node = new Node{value, nullptr};

    if (list->head == nullptr) {
        new_node->next = new_node;
        list->head = new_node;
    } else {
        Node* current = list->head;
        while (current->next != list->head)
            current = current->next;
        current->next = new_node;
        new_node->next = list->head;
    }
    list->size++;
}

// Возвращает 1 если элемент найден и удалён, 0 если не найден.
EXPORT int remove_element(void* handle, int value) {
    CyclicList* list = reinterpret_cast<CyclicList*>(handle);
    if (list->head == nullptr) return 0;

    Node* prev    = nullptr;
    Node* current = list->head;

    do {
        if (current->data == value) {
            if (prev == nullptr) {                          // удаляем head
                if (current->next == current) {            // единственный
                    list->head = nullptr;
                } else {
                    Node* last = list->head;
                    while (last->next != list->head)
                        last = last->next;
                    last->next    = list->head->next;
                    list->head    = list->head->next;
                }
            } else {
                prev->next = current->next;
            }
            delete current;
            list->size--;
            return 1;
        }
        prev    = current;
        current = current->next;
    } while (current != list->head);

    return 0;
}

EXPORT int get_size(void* handle) {
    return reinterpret_cast<CyclicList*>(handle)->size;
}

// Записывает до max_size элементов в буфер buffer (выделяется Python'ом).
EXPORT void get_elements(void* handle, int* buffer, int max_size) {
    CyclicList* list = reinterpret_cast<CyclicList*>(handle);
    if (!list->head) return;

    Node* current = list->head;
    int   i       = 0;
    do {
        if (i >= max_size) break;
        buffer[i++] = current->data;
        current     = current->next;
    } while (current != list->head);
}

EXPORT void clear_list(void* handle) {
    CyclicList* list = reinterpret_cast<CyclicList*>(handle);
    if (!list->head) return;

    Node* current = list->head;
    Node* next_node;
    do {
        next_node = current->next;
        delete current;
        current   = next_node;
    } while (current != list->head);

    list->head = nullptr;
    list->size = 0;
}
