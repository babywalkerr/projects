

REM Переходим в корень проекта
cd /d "%~dp0\.."

REM Активируем виртуальное окружение
call ".venv\Scripts\activate.bat"

REM Запускаем сервер
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

pause