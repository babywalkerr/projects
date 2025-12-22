#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>
#include <stdbool.h>

#define SIZE_MENU 7
#define GOOD_INP 1
#define ERR_INP 0

// Цвета для современного интерфейса
#define COLOR_RESET "\033[0m"
#define COLOR_CYAN "\033[96m"
#define COLOR_GREEN "\033[92m"
#define COLOR_YELLOW "\033[93m"
#define COLOR_MAGENTA "\033[95m"
#define COLOR_RED "\033[91m"
#define COLOR_BLUE "\033[94m"
#define COLOR_BOLD "\033[1m"
#define COLOR_UNDERLINE "\033[4m"
#define COLOR_BLINK "\033[5m"
#define COLOR_INVERSE "\033[7m"
#define COLOR_BG_GREEN "\033[42m"
#define COLOR_BG_RED "\033[41m"

// Структура для хранения данных интеграла
typedef struct {
    double a;          // нижняя граница
    double b;          // верхняя граница
    double step;       // шаг интегрирования
    double result;     // результат вычисления
    double accuracy;   // погрешность
    int calculated;    // флаг вычисления
} IntegralData;

// Прототипы функций
void Menu(const char** list_menu, int point);
void InputBounds(IntegralData* data);
void InputStep(IntegralData* data);
void CalculateIntegral(IntegralData* data);
double CalculateExactValue(double a, double b);
double Primitive(double x);
double FindRoot();
void DetermineAccuracy(IntegralData* data);
void PrintResult(IntegralData* data);
void AboutProgram();
double Function(double x);
void ClearInputBuffer();
void PrintHeader(const char* title);
void AnimatedProgress();

int main() {
    short is_exit = 0;
    int point = 0;

    // Подключение русского языка и расширенных цветов
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    system("chcp 1251 > nul");
    
    // Включение ANSI-цветов для Windows 10+
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);

    // Выделение памяти для меню
    const char** list_menu = malloc(SIZE_MENU * sizeof(char*));
    list_menu[0] = "Ввод границ интеграла";
    list_menu[1] = "Ввод шага интегрирования";
    list_menu[2] = "Расчёт интеграла (метод трапеций)";
    list_menu[3] = "Определить погрешность";
    list_menu[4] = "Вывод результата";
    list_menu[5] = "О программе";
    list_menu[6] = "Выход";

    // Инициализация данных интеграла
    IntegralData data = {0, 0, 0.01, 0, 0, 0};

    while (!is_exit) {
        system("cls");
        Menu(list_menu, point);

        int key = _getch();
        switch(key) {
            case 72:    // стрелка вверх
                point = (point == 0) ? SIZE_MENU - 1 : point - 1;
                break;
            case 80:    // стрелка вниз
                point = (point == SIZE_MENU - 1) ? 0 : point + 1;
                break;
            case 13:    // Enter - выбор пункта
                system("cls");
                switch(point) {
                    case 0: InputBounds(&data); break;
                    case 1: InputStep(&data); break;
                    case 2: CalculateIntegral(&data); break;
                    case 3: DetermineAccuracy(&data); break;
                    case 4: PrintResult(&data); break;
                    case 5: AboutProgram(); break;
                    case 6: is_exit = 1; break;
                }
                if (point != 6) {
                    printf("\n%s%s[Нажмите любую клавишу для возврата...]%s", 
                           COLOR_BOLD, COLOR_CYAN, COLOR_RESET);
                    _getch();
                }
                break;
            case 27:    // Escape - выход
                is_exit = 1;
                break;
        }
    }

    free(list_menu);
    printf("\n%s%s", COLOR_BOLD, COLOR_MAGENTA);
    printf("г========================================================¬\n");
    printf("¦  Программа завершена. Спасибо за использование!        ¦\n");
    printf("¦========================================================¦\n");
    printf("¦                                                        ¦\n");
    printf("¦           %s!!! Made by Maxim Epifanov !!!%s               ¦\n", COLOR_CYAN, COLOR_MAGENTA);
    printf("¦              %sVersion 11.8%s                              ¦\n", COLOR_YELLOW, COLOR_MAGENTA);
    printf("¦                                                        ¦\n");
    printf("L========================================================-\n");
    printf("%s", COLOR_RESET);
    return 0;
}

void Menu(const char** list_menu, int point) {
    printf("\n%s%s", COLOR_BOLD, COLOR_CYAN);
    printf("г========================================================¬\n");
    printf("¦                                                        ¦\n");
    printf("¦        ВЫЧИСЛЕНИЕ ОПРЕДЕЛЁННОГО ИНТЕГРАЛА              ¦\n");
    printf("¦            f(x) = 2x? - x? + 7  (y > 0)                ¦\n");
    printf("¦                                                        ¦\n");
    printf("¦              %s~ by Maxim Epifanov ~%s                     ¦\n", COLOR_YELLOW, COLOR_CYAN);
    printf("¦========================================================¦\n");
    printf("%s", COLOR_RESET);

    for (int i = 0; i < SIZE_MENU; i++) {
        if (point == i) {
            printf("%s%s¦  >>  %-46s  ¦%s\n", 
                   COLOR_BOLD, COLOR_GREEN, list_menu[i], COLOR_RESET);
        } else {
            printf("%s¦      %-46s  ¦%s\n", 
                   COLOR_CYAN, list_menu[i], COLOR_RESET);
        }
    }

    printf("%s%s¦========================================================¦\n", 
           COLOR_BOLD, COLOR_CYAN);
    printf("¦  Управление: [UP/DOWN] | [Enter] | [Esc] выход         ¦\n");
    printf("L========================================================-%s\n", 
           COLOR_RESET);
}

void PrintHeader(const char* title) {
    printf("\n%s%s", COLOR_BOLD, COLOR_MAGENTA);
    printf("г========================================================¬\n");
    printf("¦  %-52s  ¦\n", title);
    printf("L========================================================-\n");
    printf("%s\n", COLOR_RESET);
}

void InputBounds(IntegralData* data) {
    PrintHeader("ВВОД ГРАНИЦ ИНТЕГРИРОВАНИЯ");

    double a, b;
    int valid = GOOD_INP;
    const double MAX_V = 1e6, MIN_V = -1e6;
    
    // Ввод нижней границы
    do {
        valid = GOOD_INP;
        printf("%s> Введите нижнюю границу a (%.0e <= a <= %.0e): %s", 
               COLOR_CYAN, MIN_V, MAX_V, COLOR_RESET);
        
        if (scanf("%lf", &a) != 1 || getchar() != '\n') {
            ClearInputBuffer();
            printf("%s[ОШИБКА] Введите корректное число.%s\n", COLOR_RED, COLOR_RESET);
            valid = ERR_INP;
            continue;
        }
        
        if (a < MIN_V || a > MAX_V) {
            printf("%s[ОШИБКА] Значение вне допустимого диапазона!%s\n", COLOR_RED, COLOR_RESET);
            valid = ERR_INP;
        }
    } while (valid != GOOD_INP);

    // Ввод верхней границы
    do {
        valid = GOOD_INP;
        printf("%s> Введите верхнюю границу b (%.0e <= b <= %.0e): %s", 
               COLOR_CYAN, MIN_V, MAX_V, COLOR_RESET);
        
        if (scanf("%lf", &b) != 1 || getchar() != '\n') {
            ClearInputBuffer();
            printf("%s[ОШИБКА] Введите корректное число.%s\n", COLOR_RED, COLOR_RESET);
            valid = ERR_INP;
            continue;
        }
        
        if (b < MIN_V || b > MAX_V) {
            printf("%s[ОШИБКА] Значение вне допустимого диапазона!%s\n", COLOR_RED, COLOR_RESET);
            valid = ERR_INP;
            continue;
        }
        
        if (a >= b) {
            printf("%s[ОШИБКА] Верхняя граница должна быть больше нижней!%s\n", 
                   COLOR_RED, COLOR_RESET);
            valid = ERR_INP;
        }
    } while (valid != GOOD_INP);

    data->a = a;
    data->b = b;
    data->calculated = 0;

    printf("\n%s[OK] Границы успешно установлены:%s\n", COLOR_GREEN, COLOR_RESET);
    printf("  a = %s%.6lf%s\n", COLOR_YELLOW, data->a, COLOR_RESET);
    printf("  b = %s%.6lf%s\n", COLOR_YELLOW, data->b, COLOR_RESET);
    printf("  Длина интервала: %s%.6lf%s\n", COLOR_YELLOW, data->b - data->a, COLOR_RESET);
}

void InputStep(IntegralData* data) {
    PrintHeader("НАСТРОЙКА ШАГА ИНТЕГРИРОВАНИЯ");

    printf("Текущий шаг: %s%.6lf%s\n\n", COLOR_YELLOW, data->step, COLOR_RESET);
    printf("%s> Введите новый шаг (0.001 <= step <= 1): %s", COLOR_CYAN, COLOR_RESET);

    double new_step;
    char c;
    
    while (1) {
        int result = scanf("%lf%c", &new_step, &c);
        
        if (result == 2 && c == '\n') {
            if (new_step >= 0.001 && new_step <= 1) {
                data->step = new_step;
                break;
            } else {
                printf("%s[ОШИБКА] Шаг должен быть в диапазоне [0.001, 1]! Повторите: %s", 
                       COLOR_RED, COLOR_RESET);
            }
        } else {
            printf("%s[ОШИБКА] Введите число: %s", COLOR_RED, COLOR_RESET);
            ClearInputBuffer();
        }
    }
    
    data->calculated = 0;
    printf("\n%s[OK] Шаг успешно установлен: %.6lf%s\n", COLOR_GREEN, data->step, COLOR_RESET);
}

// Функция для интегрирования: f(x) = 2x? - x? + 7
double Function(double x) {
    double y = 2.0 * x * x * x - x * x + 7.0;
    return (y >= 0) ? y : 0;  // Только положительная часть
}

void AnimatedProgress() {
    printf("%s", COLOR_CYAN);
    const char* anim[] = {"|", "/", "-", "\\", "|", "/", "-", "\\"};
    for (int i = 0; i < 24; i++) {
        printf("\r[%s] Вычисление... %d%%", anim[i % 8], (i + 1) * 4);
        fflush(stdout);
        Sleep(50);
    }
    printf("%s\r                                    \r", COLOR_RESET);
}

void CalculateIntegral(IntegralData* data) {
    PrintHeader("РАСЧЁТ ИНТЕГРАЛА МЕТОДОМ ТРАПЕЦИЙ");

    if (data->a >= data->b) {
        printf("%s[ОШИБКА] Границы некорректны!%s\n", COLOR_RED, COLOR_RESET);
        return;
    }

    printf("Параметры вычисления:\n");
    printf("  • Границы: [%s%.6lf%s, %s%.6lf%s]\n", 
           COLOR_YELLOW, data->a, COLOR_RESET, 
           COLOR_YELLOW, data->b, COLOR_RESET);
    printf("  • Шаг: %s%.6lf%s\n", COLOR_YELLOW, data->step, COLOR_RESET);
    printf("  • Метод: %sТрапеций%s\n\n", COLOR_MAGENTA, COLOR_RESET);

    // Проверка на слишком большое количество шагов
    long long n = (long long)((data->b - data->a) / data->step);
    if (n < 1) n = 1;
    
    if (n > 10000000) {
        printf("%s[ПРЕДУПРЕЖДЕНИЕ] Слишком много шагов (%lld)! Это займёт много времени.%s\n", 
               COLOR_YELLOW, n, COLOR_RESET);
        printf("Рекомендуется увеличить шаг. Продолжить? (y/n): ");
        char answer;
        scanf(" %c", &answer);
        ClearInputBuffer();
        if (answer != 'y' && answer != 'Y') {
            printf("Операция отменена.\n");
            return;
        }
    }

    AnimatedProgress();

    // Метод трапеций
    double h = (data->b - data->a) / n;
    double sum = 0.0;

    // Формула трапеций: (h/2) * [f(a) + 2*sum(f(xi)) + f(b)]
    sum = Function(data->a) + Function(data->b);
    
    for (long long i = 1; i < n; i++) {
        double x = data->a + i * h;
        sum += 2.0 * Function(x);
    }
    
    data->result = (h / 2.0) * sum;
    data->calculated = 1;

    printf("\n%s[OK] Интеграл успешно вычислен!%s\n", COLOR_GREEN, COLOR_RESET);
    printf("  • Количество шагов: %s%lld%s\n", COLOR_YELLOW, n, COLOR_RESET);
    printf("  • Результат: %s%.10lf%s\n", COLOR_GREEN, data->result, COLOR_RESET);
}

// Первообразная для точного значения
double Primitive(double x) {
    return 0.5 * x * x * x * x - (x * x * x) / 3.0 + 7.0 * x;
}

// Функция для поиска корня методом бисекции: 2x? - x? + 7 = 0
double FindRoot() {
    double left = -2.0;   // Начальная левая граница
    double right = 0.0;   // Начальная правая граница
    double eps = 1e-10;   // Точность
    
    // Проверяем, что на концах разные знаки
    double f_left = 2.0 * left * left * left - left * left + 7.0;
    double f_right = 2.0 * right * right * right - right * right + 7.0;
    
    if (f_left * f_right > 0) {
        // Если знаки одинаковые, корень не в этом интервале
        return -1.4526; // Возврат к приближенному значению
    }
    
    // Метод бисекции
    while (right - left > eps) {
        double mid = (left + right) / 2.0;
        double f_mid = 2.0 * mid * mid * mid - mid * mid + 7.0;
        
        if (fabs(f_mid) < eps) {
            return mid;
        }
        
        if (f_left * f_mid < 0) {
            right = mid;
            f_right = f_mid;
        } else {
            left = mid;
            f_left = f_mid;
        }
    }
    
    return (left + right) / 2.0;
}

// Точное значение интеграла для оценки погрешности
double CalculateExactValue(double a, double b) {
    // Находим точный корень функции 2x? - x? + 7 = 0
    static double root = 0.0;
    static int calculated = 0;
    
    if (!calculated) {
        root = FindRoot();
        calculated = 1;
    }
    
    // Интегрируем только положительную часть
    double lower = (a > root) ? a : root;
    double upper = b;
    
    if (upper <= root) return 0.0;
    
    return Primitive(upper) - Primitive(lower);
}

void DetermineAccuracy(IntegralData* data) {
    PrintHeader("ОЦЕНКА ПОГРЕШНОСТИ РЕЗУЛЬТАТА");

    if (!data->calculated) {
        printf("%s[ОШИБКА] Сначала необходимо вычислить интеграл!%s\n", COLOR_RED, COLOR_RESET);
        return;
    }

    printf("Вычисленное значение (метод трапеций): %s%.10lf%s\n", 
           COLOR_CYAN, data->result, COLOR_RESET);

    AnimatedProgress();

    double exact = CalculateExactValue(data->a, data->b);
    printf("\nТочное значение интеграла: %s%.10lf%s\n", 
           COLOR_YELLOW, exact, COLOR_RESET);

    if (fabs(exact) < 1e-15) {
        printf("%s[ВНИМАНИЕ] Точное значение близко к нулю, погрешность не определена%s\n", 
               COLOR_YELLOW, COLOR_RESET);
        data->accuracy = 0.0;
    } else {
        double abs_error = fabs(data->result - exact);
        data->accuracy = (abs_error / fabs(exact)) * 100.0;
        
        printf("\n%sАнализ точности:%s\n", COLOR_BOLD, COLOR_RESET);
        printf("  • Абсолютная погрешность: %s%.10lf%s\n", 
               COLOR_CYAN, abs_error, COLOR_RESET);
        printf("  • Относительная погрешность: %s%.6lf%%%s\n", 
               COLOR_MAGENTA, data->accuracy, COLOR_RESET);
        
        if (data->accuracy < 0.01) {
            printf("  • Оценка: %s[ОТЛИЧНО] Высокая точность%s\n", COLOR_GREEN, COLOR_RESET);
        } else if (data->accuracy < 1.0) {
            printf("  • Оценка: %s[ХОРОШО] Приемлемая точность%s\n", COLOR_YELLOW, COLOR_RESET);
        } else {
            printf("  • Оценка: %s[УДОВЛЕТВОРИТЕЛЬНО] Требуется меньший шаг%s\n", COLOR_RED, COLOR_RESET);
        }
    }
}

void PrintResult(IntegralData* data) {
    PrintHeader("ИТОГОВЫЕ РЕЗУЛЬТАТЫ");

    printf("%s", COLOR_BOLD);
    printf("г========================================================¬\n");
    printf("¦  ПАРАМЕТРЫ ИНТЕГРИРОВАНИЯ                              ¦\n");
    printf("¦========================================================¦\n");
    printf("%s", COLOR_RESET);
    printf("¦  Функция: f(x) = 2x? - x? + 7  (при y > 0)             ¦\n");
    printf("¦  Нижняя граница: a = %-33.6lf¦\n", data->a);
    printf("¦  Верхняя граница: b = %-32.6lf¦\n", data->b);
    printf("¦  Длина интервала: %-37.6lf¦\n", data->b - data->a);
    printf("¦  Шаг интегрирования: %-34.6lf¦\n", data->step);
    printf("%s¦========================================================¦\n", COLOR_BOLD);
    printf("%s", COLOR_RESET);

    if (data->calculated) {
        printf("%s¦  РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ                                 ¦\n", COLOR_BOLD);
        printf("¦========================================================¦\n");
        printf("%s", COLOR_RESET);
        printf("¦  Метод: Трапеций                                       ¦\n");
        printf("¦  Значение интеграла: %-30.10lf¦\n", data->result);
        
        if (data->accuracy > 0) {
            printf("¦  Относительная погрешность: %-23.6lf%%¦\n", data->accuracy);
        }
    } else {
        printf("¦  [!] ИНТЕГРАЛ ЕЩЁ НЕ ВЫЧИСЛЕН                          ¦\n");
        printf("¦  Выберите пункт \"Расчёт интеграла\"                      ¦\n");
    }

    printf("%sL========================================================-\n", COLOR_BOLD);
    printf("%s", COLOR_RESET);
}

void AboutProgram() {
    PrintHeader("ИНФОРМАЦИЯ О ПРОГРАММЕ");

    printf("%s", COLOR_BOLD);
    printf("г========================================================¬\n");
    printf("¦  ПРОГРАММА ЧИСЛЕННОГО ИНТЕГРИРОВАНИЯ                   ¦\n");
    printf("¦========================================================¦\n");
    printf("%s", COLOR_RESET);
    printf("¦                                                        ¦\n");
    printf("¦  Назначение:                                           ¦\n");
    printf("¦     Вычисление площади фигуры, ограниченной            ¦\n");
    printf("¦     кривой и осью OX в положительной области           ¦\n");
    printf("¦                                                        ¦\n");
    printf("¦  Функция:                                              ¦\n");
    printf("¦     f(x) = 2x? - x? + 7                                ¦\n");
    printf("¦                                                        ¦\n");
    printf("¦  Метод:                                                ¦\n");
    printf("¦     Численное интегрирование методом трапеций          ¦\n");
    printf("¦                                                        ¦\n");
    printf("¦  Возможности:                                          ¦\n");
    printf("¦     • Настройка границ интегрирования                  ¦\n");
    printf("¦     • Регулировка шага вычислений                      ¦\n");
    printf("¦     • Оценка погрешности результата                    ¦\n");
    printf("¦     • Работа только с положительной частью функции     ¦\n");
    printf("¦                                                        ¦\n");
    printf("¦  Управление:                                           ¦\n");
    printf("¦     [UP]/[DOWN]  - навигация по меню                   ¦\n");
    printf("¦     [Enter]      - выбор пункта                        ¦\n");
    printf("¦     [Esc]        - выход из программы                  ¦\n");
    printf("¦                                                        ¦\n");
    printf("%s¦========================================================¦\n", COLOR_BOLD);
    printf("¦                                                        ¦\n");
    printf("¦           %s!!! Made by Maxim Epifanov !!!%s               ¦\n", COLOR_CYAN, COLOR_BOLD);
    printf("¦              %sVersion 11.8%s                              ¦\n", COLOR_YELLOW, COLOR_BOLD);
    printf("¦          %sЛабораторная работа №3 • 2025%s                 ¦\n", COLOR_MAGENTA, COLOR_BOLD);
    printf("¦                                                        ¦\n");
    printf("L========================================================-\n");
    printf("%s", COLOR_RESET);
}

void ClearInputBuffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {}
}