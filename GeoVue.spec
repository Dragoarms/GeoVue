# -*- mode: python ; coding: utf-8 -*-
import os
import sys
# bump the recursion limit to 5× the default
sys.setrecursionlimit(sys.getrecursionlimit() * 5)
from glob import glob
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# -------------------------------------------------------------------
# Manually gather every resource under src/resources → resources/...
# -------------------------------------------------------------------
datas = [
    # top‐level files
    ("src/resources/FolderIcons.png",        "resources"),
    ("src/resources/full_logo.png",          "resources"),
    ("src/resources/logo.ico",               "resources"),
    ("src/resources/translations.csv",       "resources"),
]

# all ArUco SVGs
datas += [
    (svg, os.path.join("resources", "ArUco Markers"))
    for svg in glob("src/resources/ArUco Markers/*.svg")
]

# all color‐preset JSONs
datas += [
    (jsn, os.path.join("resources", "color_presets"))
    for jsn in glob("src/resources/color_presets/*.json")
]

# all folder-icon ICOs
datas += [
    (ico, os.path.join("resources", "Icons"))
    for ico in glob("src/resources/Icons/*.ico")
]

# the single Register‐template
datas += [
    (
        "src/resources/Register Template File/Chip_Tray_Register.xltx",
        os.path.join("resources", "Register Template File")
    )
]

# config .json file
datas += [
    ("src/config.json", "_internal"),  # ← this line ensures dist/GeoVue/_internal/config.json exists
]

# -------------------------------------------------------------------
# Analysis: collects your Python code, modules, and the datas above
# -------------------------------------------------------------------
a = Analysis(
    ["launcher.py"],           # your entry‐point script
    pathex=["src"],            # where to find launcher.py & other modules
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "opencv_videoio_ffmpeg4100_64",
        "torch_cpu",
        "torch",
        "torchvision",
        "sphinx",
        "black",
        "jupyter",
        "Ipython"
    ],
)

# -------------------------------------------------------------------
# Build the Python byte-code archive
# -------------------------------------------------------------------
pyz = PYZ(
    a.pure,
    a.zipped_data,
)

# -------------------------------------------------------------------
# Build the EXE (one-folder mode)
# -------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],                      # no binaries packaged here
    exclude_binaries=True,   # DLLs & .pyd go into COLLECT
    name="GeoVue",
    debug=False,
    icon="src/resources/logo.ico",
    console=False,
)

# -------------------------------------------------------------------
# Collect everything into dist/GeoVue/
# -------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="GeoVue",
)
