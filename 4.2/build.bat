@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo  Lab 4.2 - Building C++ modules
echo  Working dir: %CD%
echo ============================================================
echo.

:: Check that modules folder exists
if not exist "modules" (
    echo ERROR: 'modules' folder not found in %CD%
    echo Make sure build.bat is inside the lab4_2 root folder.
    pause
    exit /b 1
)

if not exist "modules\cyclic_list.cpp" (
    echo ERROR: modules\cyclic_list.cpp not found
    pause
    exit /b 1
)
if not exist "modules\cyclic_list_stl.cpp" (
    echo ERROR: modules\cyclic_list_stl.cpp not found
    pause
    exit /b 1
)

:: Build 1: dynamic structs
echo [1/2] Compiling cyclic_list.dll (dynamic structs)...
g++ -shared -o modules\cyclic_list.dll modules\cyclic_list.cpp -std=c++17

if errorlevel 1 (
    echo   FAILED: cyclic_list.dll
) else (
    echo   OK: modules\cyclic_list.dll
)

:: Build 2: STL
echo.
echo [2/2] Compiling cyclic_list_stl.dll (STL)...
g++ -shared -o modules\cyclic_list_stl.dll modules\cyclic_list_stl.cpp -std=c++17

if errorlevel 1 (
    echo   FAILED: cyclic_list_stl.dll
) else (
    echo   OK: modules\cyclic_list_stl.dll
)

echo.
echo ============================================================
echo  Done. Run: python main.py
echo ============================================================
pause