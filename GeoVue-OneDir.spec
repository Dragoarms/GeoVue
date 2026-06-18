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

# Refresh resource data from files that actually exist.
datas = []
resource_root = os.path.join("src", "resources")
for root, dirnames, filenames in os.walk(resource_root):
    dirnames[:] = [dirname for dirname in dirnames if dirname != "__pycache__"]
    relative_dir = os.path.relpath(root, resource_root)
    target_dir = "resources" if relative_dir == "." else os.path.join("resources", relative_dir)

    for filename in filenames:
        if filename.endswith((".py", ".pyc", ".pyo")):
            continue
        datas.append((os.path.join(root, filename), target_dir))

# config .json file
datas += [
    ("src/config.json", "."),  # ← this line ensures dist/GeoVue/_internal/config.json exists
]

# -------------------------------------------------------------------
# Analysis: collects your Python code, modules, and the datas above
# -------------------------------------------------------------------
a = Analysis(
    ["launcher.py"],           # your entry‐point script
    pathex=["src"],            # where to find launcher.py & other modules
    binaries=[],
    datas=datas,
    hiddenimports=[
        # snowflake-connector-python (externalbrowser SSO auth)
        "snowflake.connector",
        "snowflake.connector.auth",
        "snowflake.connector.auth_default",
        "snowflake.connector.auth_webbrowser",
        "snowflake.connector.connection",
        "snowflake.connector.cursor",
        "snowflake.connector.errors",
        "snowflake.connector.network",
        "snowflake.connector.ssl_wrap_socket",
        "snowflake.connector.vendored",
        "snowflake.connector.vendored.urllib3",
        "snowflake.connector.vendored.urllib3.contrib",
        "snowflake.connector.vendored.urllib3.contrib.pyopenssl",
        # crypto dependencies
        "cryptography",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.asymmetric",
        "cryptography.hazmat.backends",
        "cryptography.hazmat.backends.openssl",
        "oscrypto",
        "asn1crypto",
        "pyOpenSSL",
        "OpenSSL",
        "OpenSSL.SSL",
        # parquet cache
        "pyarrow",
        "pyarrow.pandas_compat",
    ],
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
    debug=True,
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
