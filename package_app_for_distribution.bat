@echo off
setlocal EnableDelayedExpansion

:: Extract version from src/config.json using PowerShell
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "(Get-Content 'src\config.json' | ConvertFrom-Json).version"`) do (
    set "VERSION=%%a"
)

if "%VERSION%"=="" (
    echo ERROR: Could not extract version from src/config.json
    echo Make sure src/config.json contains a "version" field
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

set "DIST_NAME=GeoVue_v%VERSION%_Installer"

echo ========================================
echo Building GeoVue v%VERSION%...
echo ========================================

:: Clean and build
if exist dist rmdir /s /q dist
pyinstaller GeoVue-OneDir.spec

if %errorLevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

:: Create distribution package
echo.
echo Creating distribution package...
if exist "%DIST_NAME%" rmdir /s /q "%DIST_NAME%"
mkdir "%DIST_NAME%"

:: Copy files
xcopy dist "%DIST_NAME%\dist\" /E /I /Y /Q
copy install.bat "%DIST_NAME%\" >nul
copy src\config.json "%DIST_NAME%\" >nul

:: Create English README (batch can handle this)
(
echo ========================================
echo GeoVue v%VERSION% - Chip Tray Photos
echo ========================================
echo.
echo INSTALLATION INSTRUCTIONS
echo -------------------------
echo 1. Run install.bat
echo    - This will install GeoVue to your user AppData folder
echo    - No administrator privileges required
echo.
echo 2. Launch GeoVue from:
echo    - Desktop shortcut
echo    - Start Menu -^> GeoVue
echo.
echo 3. First Run Setup:
echo    - On first launch, GeoVue will guide you through setup
echo    - Please use the default local working folder ^(C:\GeoVue Chip Tray Photos^) this will ensure that you can work locally without network issues. HOWEVER, after you finish working, in the main interface - click 'Sync to Cloud' which will automatically attempt to upload all your work to the correct locations in teams. Files will be tagged with 'UPLOADED' once they're confirmed uploaded to teams successfully. There should be an automatic clean up of these files but i haven't tested it yet.
echo    - In the initial dialog under the shared folder path - select 'Exploration Drilling / 03 - Reverse Circulation / Chip Tray Photos folder.
echo    - The wizard will create all required folder structures locally and map itself to the shared teams folder.
echo.
echo 	NOTE
echo You may need to synchronize the chip tray photos folder in teams before it is visible in your folder directory.
) > "%DIST_NAME%\README.txt"

:: Create French README
(
echo ========================================
echo GeoVue v%VERSION% - Photos de plateaux de copeaux
echo ========================================
echo.
echo INSTRUCTIONS D'INSTALLATION
echo ---------------------------
echo 1. Executez install.bat
echo    - Ceci installera GeoVue dans votre dossier AppData utilisateur
echo    - Aucun privilege administrateur requis
echo.
echo 2. Lancez GeoVue depuis :
echo    - Raccourci du bureau
echo    - Menu Demarrer -^> GeoVue
echo.
echo 3. Configuration de la premiere execution :
echo    - Au premier lancement, GeoVue vous guidera dans la configuration
echo    - Utilisez le dossier de travail local par defaut ^(C:\GeoVue Chip Tray Photos^) pour travailler localement sans probleme de reseau. Quand vous avez termine, cliquez sur 'Sync to Cloud' dans l'interface principale pour telecharger votre travail vers les bons emplacements Teams.
echo    - Dans la boite de dialogue initiale, sous le chemin du dossier partage, selectionnez 'Exploration Drilling / 03 - Reverse Circulation / Chip Tray Photos'.
echo    - L'assistant creera les dossiers requis localement et les associera au dossier Teams partage.
echo.
echo 	NOTE
echo Vous devrez peut-etre synchroniser le dossier Chip Tray Photos dans Teams avant qu'il soit visible dans votre explorateur de fichiers.
) > "%DIST_NAME%\README_FR.txt"
goto :after_french_readme

:: Legacy French README block kept unreachable for reference
powershell -NoProfile -Command ^
"$text = '========================================'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8; ^
$text = 'GeoVue v%VERSION% - Photos de plateaux de copeaux'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '========================================'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = ''; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = 'INSTRUCTIONS D''INSTALLATION'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '---------------------------'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '1. Exécutez install.bat'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - Ceci installera GeoVue dans votre dossier AppData utilisateur'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - Aucun privilège d''administrateur requis'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = ''; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '2. Lancez GeoVue depuis :'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - Raccourci du bureau'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - Menu Démarrer → GeoVue'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = ''; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '3. Configuration de la première exécution :'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - Au premier lancement, GeoVue vous guidera dans la configuration'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - Veuillez utiliser le dossier de travail local par défaut (C:\GeoVue Chip Tray Photos) cela garantira que vous pouvez travailler localement sans problèmes de réseau. CEPENDANT, après avoir terminé votre travail, dans l''interface principale - cliquez sur ''Sync to Cloud'' qui tentera automatiquement de télécharger tout votre travail aux bons emplacements dans Teams. Les fichiers seront marqués ''UPLOADED'' une fois qu''ils sont confirmés téléchargés avec succès dans Teams. Il devrait y avoir un nettoyage automatique de ces fichiers mais je ne l''ai pas encore testé.'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - Dans la boîte de dialogue initiale sous le chemin du dossier partagé - sélectionnez ''Exploration Drilling / 03 - Reverse Circulation / Chip Tray Photos''.'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '   - L''assistant créera toutes les structures de dossiers requises localement et se mappera au dossier Teams partagé.'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = ''; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = '	NOTE'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append; ^
$text = 'Vous devrez peut-être synchroniser le dossier Chip Tray Photos dans Teams avant qu''il ne soit visible dans votre répertoire de dossiers.'; $text | Out-File -FilePath '%DIST_NAME%\README_FR.txt' -Encoding UTF8 -Append"

echo.
echo ========================================
:after_french_readme
echo Package created: %DIST_NAME%
echo.
echo Contents:
echo - install.bat (installer script)
echo - config.json (version info)
echo - README.txt (English instructions)
echo - README_FR.txt (French instructions)
echo - dist\GeoVue\ (application files)
echo.
echo You can now zip this folder for distribution
echo ========================================
if not "%NO_PAUSE%"=="1" pause
