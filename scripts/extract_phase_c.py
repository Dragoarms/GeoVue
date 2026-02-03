"""Extract Phase C modules (charts, tables, images, collar_map) from logging_review_html_report.py."""
import os

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(repo, "src", "processing", "logging_review_html_report.py")
html_dir = os.path.join(repo, "src", "reports", "logging_review", "html")

with open(html_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def find_def(lines, name):
    for i, L in enumerate(lines):
        if L.strip().startswith("def " + name + "("):
            return i
    return None

def find_next_def(lines, start, exclude_names):
    for i in range(start + 1, len(lines)):
        L = lines[i]
        if L.strip().startswith("def "):
            return i
    return len(lines)

# Charts: 8 _plotly_* functions
chart_start = find_def(lines, "_plotly_pie_json")
chart_end = find_next_def(lines, find_def(lines, "_plotly_outlier_scatter_json"), [])
charts_content = "".join(lines[chart_start:chart_end])

charts_py = os.path.join(html_dir, "charts.py")
with open(charts_py, "w", encoding="utf-8") as f:
    f.write('"""Plotly chart JSON generators for logging review HTML report."""\n')
    f.write("import json\n")
    f.write("from typing import Any, Dict, List, Optional, Tuple\n\n")
    f.write("import pandas as pd\n\n")
    f.write(charts_content)
print("Wrote", charts_py)

# Tables: 7 _render_* table functions
table_start = find_def(lines, "_render_intervals_table")
table_end = find_next_def(lines, find_def(lines, "_render_grouping_groups"), [])
tables_content = "".join(lines[table_start:table_end])

tables_py = os.path.join(html_dir, "tables.py")
with open(tables_py, "w", encoding="utf-8") as f:
    f.write('"""Table renderers for logging review HTML report."""\n')
    f.write("import html\n")
    f.write("from typing import Any, Dict, List\n\n")
    f.write(tables_content)
print("Wrote", tables_py)

# Images: _encode_image_base64 only
img_start = find_def(lines, "_encode_image_base64")
img_end = find_next_def(lines, img_start, [])
images_content = "".join(lines[img_start:img_end])

images_py = os.path.join(html_dir, "images.py")
with open(images_py, "w", encoding="utf-8") as f:
    f.write('"""Image encoding for logging review HTML report."""\n')
    f.write("import base64\n")
    f.write("import logging\n")
    f.write("import os\n")
    f.write("from pathlib import Path\n")
    f.write("from typing import Optional\n\n")
    f.write("logger = logging.getLogger(__name__)\n\n")
    f.write(images_content)
print("Wrote", images_py)

# Collar map: _build_map_points and _render_map
build_start = find_def(lines, "_build_map_points")
build_end = find_next_def(lines, build_start, [])
render_map_start = find_def(lines, "_render_map")
render_map_end = find_next_def(lines, render_map_start, [])
collar_content = "".join(lines[build_start:build_end]) + "\n\n" + "".join(lines[render_map_start:render_map_end])

collar_py = os.path.join(html_dir, "collar_map.py")
with open(collar_py, "w", encoding="utf-8") as f:
    f.write('"""Collar map building and rendering for logging review HTML report."""\n')
    f.write("import json\n")
    f.write("import html\n")
    f.write("import logging\n")
    f.write("from typing import Any, Dict, Optional, Set\n\n")
    f.write("import numpy as np\n")
    f.write("import pandas as pd\n\n")
    f.write("from processing.DataManager.column_aliases import ColumnResolver\n\n")
    f.write("logger = logging.getLogger(__name__)\n\n")
    f.write("try:\n    import pyproj\n    PYPROJ_AVAILABLE = True\nexcept ImportError:\n    PYPROJ_AVAILABLE = False\n\n")
    f.write("def _resolve_coordinate_columns(df):\n")
    f.write('    """Resolve easting/northing column names using ColumnResolver."""\n')
    f.write("    resolver = ColumnResolver(df)\n")
    f.write('    return resolver.get("easting"), resolver.get("northing")\n\n')
    f.write(collar_content)
print("Wrote", collar_py)
