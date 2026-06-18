@echo off
setlocal EnableDelayedExpansion

:: Extract version from config.json
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "(Get-Content 'config.json' | ConvertFrom-Json).version"`) do (
    set "VERSION=%%a"
)

if "%VERSION%"=="" (
    echo ERROR: Could not extract version from config.json
    pause
    exit /b 1
)

:: Configuration
set "APP_NAME=GeoVue"
set "APP_NAME_WITH_VERSION=%APP_NAME% v%VERSION%"
set "EXE_NAME=GeoVue.exe"
set "SOURCE_FOLDER=%~dp0dist\GeoVue"

:: Installation paths (Programs folder, not data folder!)
set "INSTALL_DIR=%LOCALAPPDATA%\Programs\%APP_NAME%"
set "START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs\%APP_NAME%"
set "DESKTOP=%USERPROFILE%\Desktop"

cls
echo ========================================
echo %APP_NAME_WITH_VERSION% Installer
echo ========================================
echo.

:: Check if source exists
if not exist "%SOURCE_FOLDER%\%EXE_NAME%" (
    echo ERROR: Application files not found at %SOURCE_FOLDER%
    echo Please ensure dist\GeoVue\ folder exists
    pause
    exit /b 1
)

:: Check if already installed
if exist "%INSTALL_DIR%\%EXE_NAME%" (
    echo %APP_NAME% is already installed
    echo.
    choice /C YN /M "Reinstall/Update"
    if !errorLevel! equ 2 exit /b 0
    echo.
    echo Removing old installation...
    rmdir /S /Q "%INSTALL_DIR%" 2>nul
    timeout /t 1 >nul
)

:: Create installation directory
echo Installing to: %INSTALL_DIR%
mkdir "%INSTALL_DIR%" 2>nul

:: Copy application files
echo Copying application files...
xcopy "%SOURCE_FOLDER%\*" "%INSTALL_DIR%\" /E /I /Y /Q >nul
if %errorLevel% neq 0 (
    echo ERROR: Failed to copy files
    pause
    exit /b 1
)

:: Create version file for reference
echo %VERSION% > "%INSTALL_DIR%\version.txt"

:: Create Start Menu folder
if not exist "%START_MENU%" mkdir "%START_MENU%"

:: Create Desktop shortcut
echo Creating shortcuts...
powershell -NoProfile -Command ^
"$ws = New-Object -COM WScript.Shell; ^
$s = $ws.CreateShortcut('%DESKTOP%\%APP_NAME%.lnk'); ^
$s.TargetPath = '%INSTALL_DIR%\%EXE_NAME%'; ^
$s.WorkingDirectory = '%INSTALL_DIR%'; ^
$s.Description = '%APP_NAME_WITH_VERSION%'; ^
$s.Save()"

:: Create Start Menu shortcut
powershell -NoProfile -Command ^
"$ws = New-Object -COM WScript.Shell; ^
$s = $ws.CreateShortcut('%START_MENU%\%APP_NAME%.lnk'); ^
$s.TargetPath = '%INSTALL_DIR%\%EXE_NAME%'; ^
$s.WorkingDirectory = '%INSTALL_DIR%'; ^
$s.Description = '%APP_NAME_WITH_VERSION%'; ^
$s.Save()"

:: Create uninstaller
echo @echo off > "%INSTALL_DIR%\uninstall.bat"
echo echo Uninstalling %APP_NAME%... >> "%INSTALL_DIR%\uninstall.bat"
echo cd /d "%%TEMP%%" >> "%INSTALL_DIR%\uninstall.bat"
echo rmdir /S /Q "%INSTALL_DIR%" >> "%INSTALL_DIR%\uninstall.bat"
echo del "%DESKTOP%\%APP_NAME%.lnk" 2^>nul >> "%INSTALL_DIR%\uninstall.bat"
echo rmdir /S /Q "%START_MENU%" 2^>nul >> "%INSTALL_DIR%\uninstall.bat"
echo echo Note: User data in "C:\GeoVue Chip Tray Photos" was NOT removed >> "%INSTALL_DIR%\uninstall.bat"
echo echo Delete it manually if needed >> "%INSTALL_DIR%\uninstall.bat"
echo pause >> "%INSTALL_DIR%\uninstall.bat"

:: Create Start Menu uninstall shortcut
powershell -NoProfile -Command ^
"$ws = New-Object -COM WScript.Shell; ^
$s = $ws.CreateShortcut('%START_MENU%\Uninstall %APP_NAME%.lnk'); ^
$s.TargetPath = '%INSTALL_DIR%\uninstall.bat'; ^
$s.WorkingDirectory = '%INSTALL_DIR%'; ^
$s.Description = 'Uninstall %APP_NAME_WITH_VERSION%'; ^
$s.Save()"

:: Success message
echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Version: %VERSION%
echo Location: %INSTALL_DIR%
echo.
echo Shortcuts created:
echo - Desktop
echo - Start Menu
echo.
echo TO PIN TO TASKBAR:
echo 1. Click Start Menu
echo 2. Find %APP_NAME%
echo 3. Right-click and select "Pin to taskbar"
echo.
echo The app will create its data folders on first run.
echo ========================================
pause