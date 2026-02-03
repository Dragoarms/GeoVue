"""One-off: extract CSS and JS from logging_review_html_report.py into assets."""
import os

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(repo, "src", "processing", "logging_review_html_report.py")
assets_dir = os.path.join(repo, "src", "reports", "logging_review", "html", "assets")

with open(html_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# CSS: content between "    <style>" and "    </style>"
css_start = next(i for i, L in enumerate(lines) if "    <style>" in L and "</style>" not in L)
css_end = next(i for i, L in enumerate(lines) if i > css_start and "    </style>" in L)
css_lines = lines[css_start + 1 : css_end]
css_text = "".join(css_lines).replace("{{", "{").replace("}}", "}")

styles_path = os.path.join(assets_dir, "styles.py")
with open(styles_path, "w", encoding="utf-8") as out:
    out.write('"""CSS for logging review HTML report."""\n\n')
    out.write('CSS_STYLES: str = """')
    out.write(css_text)
    out.write('"""\n')
print("Wrote", styles_path, "(%d CSS lines)" % len(css_lines))

# JS: content between "    <script>" (inline) and "    </script>"
script_starts = [i for i, L in enumerate(lines) if L.strip() == "<script>"]
script_ends = [i for i, L in enumerate(lines) if L.strip() == "</script>"]
# Use the block that has substantial JS (the one after </style> and before final </script>)
js_start = next(i for i, L in enumerate(lines) if i > css_end and "    <script>" in L and "leaflet" not in "".join(lines[max(0,i-2):i+2]).lower())
js_end = next(i for i, L in enumerate(lines) if i > js_start and L.strip() == "</script>")
js_lines = lines[js_start + 1 : js_end]
js_text = "".join(js_lines).replace("{{", "{").replace("}}", "}")

scripts_path = os.path.join(assets_dir, "scripts.py")
with open(scripts_path, "w", encoding="utf-8") as out:
    out.write('"""JavaScript for logging review HTML report."""\n\n')
    out.write('JS_SCRIPTS: str = """')
    out.write(js_text)
    out.write('"""\n')
print("Wrote", scripts_path, "(%d JS lines)" % len(js_lines))
