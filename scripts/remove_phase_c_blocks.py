"""Remove Phase C function definitions from logging_review_html_report.py (now in reports.*)."""
import os

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(repo, "src", "processing", "logging_review_html_report.py")

with open(html_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def find_def(lines, name):
    for i, L in enumerate(lines):
        if L.strip().startswith("def " + name + "("):
            return i
    return None

def find_next_def(lines, start):
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("def "):
            return i
    return len(lines)

# Remove from bottom to top so indices don't shift
# 1. _render_map
# 2. _plotly_pie_json through _plotly_outlier_scatter_json
# 3. _build_map_points
# 4. _encode_image_base64
# 5. _render_intervals_table through _render_grouping_groups

ranges_to_remove = []
# _render_map
s = find_def(lines, "_render_map")
e = find_next_def(lines, s)
ranges_to_remove.append((s, e))
# charts (all 8 _plotly_*)
s = find_def(lines, "_plotly_pie_json")
e = find_def(lines, "_render_map")  # end before _render_map
ranges_to_remove.append((s, e))
# _build_map_points
s = find_def(lines, "_build_map_points")
e = find_next_def(lines, s)
ranges_to_remove.append((s, e))
# _encode_image_base64
s = find_def(lines, "_encode_image_base64")
e = find_next_def(lines, s)
ranges_to_remove.append((s, e))
# tables (all 7)
s = find_def(lines, "_render_intervals_table")
e = find_def(lines, "_plotly_pie_json")  # end before _plotly_pie_json
ranges_to_remove.append((s, e))

# Build set of line indices to remove (0-based)
remove_indices = set()
for start, end in ranges_to_remove:
    for i in range(start, end):
        remove_indices.add(i)
new_lines = [L for i, L in enumerate(lines) if i not in remove_indices]

with open(html_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Removed", len(ranges_to_remove), "blocks (%d lines)" % len(remove_indices), "from", html_path)
