@echo off
echo Starting GeoVue Validation Tool...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Install required packages if missing
echo Checking dependencies...
python -c "import PIL, pandas, tkinter" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install Pillow pandas
)

REM Run the validation tool
echo.
echo Launching validation tool...
python validation_tool.py

if errorlevel 1 (
    echo.
    echo ERROR: Validation tool failed to start
    pause
)