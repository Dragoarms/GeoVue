"""Replace inline <style> and <script> blocks with CSS_STYLES and JS_SCRIPTS in HTML report."""
import os

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(repo, "src", "processing", "logging_review_html_report.py")

with open(html_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find style block: line with "    <style>" (only) then "    </style>"
style_start = next(i for i, L in enumerate(lines) if L.rstrip() == "    <style>")
style_end = next(i for i, L in enumerate(lines) if i > style_start and L.rstrip() == "    </style>")

# Find inline script block (the one after </style>, not the Leaflet/Plotly script tags)
# It's the <script> that has no "src=" on same or next line
script_start = None
for i, L in enumerate(lines):
    if i <= style_end:
        continue
    if L.rstrip() == "    <script>":
        script_start = i
        break
script_end = next(i for i, L in enumerate(lines) if i > script_start and L.rstrip() == "    </script>")

# Build new content
replacement_style = '    <style>\n""" + CSS_STYLES + """\n    </style>'
replacement_script = '    <script>\n""" + JS_SCRIPTS + """\n    </script>'

new_lines = (
    lines[: style_start]
    + [replacement_style + "\n"]
    + lines[style_end + 1 : script_start]
    + [replacement_script + "\n"]
    + lines[script_end + 1 :]
)

with open(html_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Replaced style block (lines %d-%d) and script block (lines %d-%d)" % (style_start + 1, style_end + 1, script_start + 1, script_end + 1))
