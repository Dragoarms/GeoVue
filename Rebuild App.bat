@echo off
echo ========================================
echo GeoVue Build Options
echo ========================================
echo.
echo 1. Build ONE-FILE with NO UPX (Recommended first try)
echo 2. Build ONE-FOLDER (Fastest startup)
echo 3. Build with CONSOLE (for debugging)
echo 4. Clean all build files
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto onefile_noupx
if "%choice%"=="2" goto onefolder
if "%choice%"=="3" goto debug
if "%choice%"=="4" goto clean
goto end

:onefile_noupx
echo.
echo Building ONE-FILE without UPX compression...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller GeoVue.spec
echo.
echo Build complete! Check dist\ChipTrayApp.exe
goto end

:onefolder
echo.
echo Building ONE-FOLDER distribution...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller GeoVue-OneDir.spec
echo.
echo Build complete! Check dist\GeoVue\ folder
echo You can create a shortcut to GeoVue.exe
goto end

:debug
echo.
echo Building with CONSOLE for debugging...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller --onefile --console --name GeoVue-Debug launcher.py
echo.
echo Build complete! Run dist\GeoVue-Debug.exe to see console output
goto end

:clean
echo.
echo Cleaning all build files...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist __pycache__ rmdir /s /q __pycache__
if exist src\__pycache__ rmdir /s /q src\__pycache__
for /r %%i in (*.pyc) do del "%%i"
echo Clean complete!
goto end

:end
echo.
pause