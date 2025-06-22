# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

# Explicitly include your config and translation files
datas = [
    (os.path.abspath("src/config.json"), "."),  # will be placed in the root at runtime
    (os.path.abspath("src/resources/translations.csv"), "resources"),  # into a 'resources/' folder
    (os.path.abspath("src/resources/color_presets"), "resources/color_presets"),  # Include entire color_presets folder
    (os.path.abspath("src/resources/Register Template File/Chip_Tray_Register.xltx"), "resources/Register Template File"),    
]

a = Analysis(
    ['src\\main.py'],
    pathex=[],
    binaries=[],
    datas=datas,  # <- now including your files
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ChipTrayApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['src\\resources\\logo.ico'],
)
