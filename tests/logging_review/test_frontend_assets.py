import shutil
import subprocess

import pytest

from reports.logging_review.html.assets.scripts import JS_SCRIPTS
from reports.logging_review.html.assets.styles import CSS_STYLES
from reports.logging_review.html.tables import _render_mineralisation_evidence_table


def _extract_js_function(source: str, name: str) -> str:
    marker = f"function {name}"
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        ch = source[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    raise AssertionError(f"Could not extract {name}")


def _run_node(script: str) -> str:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


DOM_HARNESS = r"""
class ClassList {
  constructor(initial) { this.items = new Set(initial || []); }
  add(...names) { names.forEach(n => this.items.add(n)); }
  remove(...names) { names.forEach(n => this.items.delete(n)); }
  contains(name) { return this.items.has(name); }
  toggle(name, force) {
    if (force === undefined) {
      if (this.items.has(name)) { this.items.delete(name); return false; }
      this.items.add(name); return true;
    }
    if (force) this.items.add(name); else this.items.delete(name);
    return force;
  }
}
class Cell {
  constructor(text, sort) {
    this.textContent = String(text || '');
    this.dataset = {};
    if (sort !== undefined) this.dataset.sort = String(sort);
  }
}
class Header extends Cell {
  constructor(cellIndex, text) {
    super(text || '');
    this.cellIndex = cellIndex;
    this.classList = new ClassList(['sortable']);
    this.listeners = {};
  }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  click() { this.listeners.click(); }
}
class Row {
  constructor(cells, classes) {
    this.children = cells;
    this.classList = new ClassList(classes || []);
  }
}
class TBody {
  constructor(rows) { this.rows = rows; }
  querySelectorAll(selector) {
    if (selector === 'tr') return this.rows.slice();
    if (selector === 'tr:not(.all-issues-hole-header)') {
      return this.rows.filter(row => !row.classList.contains('all-issues-hole-header'));
    }
    throw new Error('unsupported selector ' + selector);
  }
  appendChild(row) {
    const idx = this.rows.indexOf(row);
    if (idx >= 0) this.rows.splice(idx, 1);
    this.rows.push(row);
  }
}
class Table {
  constructor(id, headers, rows) {
    this.id = id;
    this.headers = headers;
    this.tbody = new TBody(rows);
  }
  querySelectorAll(selector) {
    if (selector === 'th.sortable') return this.headers;
    return [];
  }
  querySelector(selector) {
    if (selector === 'tbody') return this.tbody;
    return null;
  }
}
function makeRow(valuesByIndex) {
  const cells = [];
  for (let i = 0; i < 14; i++) cells.push(new Cell('', ''));
  Object.entries(valuesByIndex).forEach(([idx, value]) => {
    cells[Number(idx)] = new Cell(value, value);
  });
  return new Row(cells);
}
"""


def test_sortable_tables_use_true_cell_index_for_noncontiguous_outlier_columns():
    init_sortable = _extract_js_function(JS_SCRIPTS, "initSortableTables")
    script = DOM_HARNESS + init_sortable + r"""
const sortableHeaders = [1, 2, 3, 4, 8, 9, 10, 11].map(i => new Header(i, 'h' + i));
function exercise(headerOffset, colIndex) {
  const rows = [
    makeRow({5: 'AAA', [colIndex]: 3}),
    makeRow({5: 'BBB', [colIndex]: 1}),
    makeRow({5: 'CCC', [colIndex]: 2}),
  ];
  const table = new Table('outlier-evidence-table', sortableHeaders, rows);
  global.document = { querySelectorAll: sel => sel === '.sortable-table' ? [table] : [] };
  initSortableTables();
  sortableHeaders[headerOffset].click();
  const ordered = table.tbody.rows.map(row => row.children[colIndex].dataset.sort).join(',');
  if (ordered !== '1,2,3') throw new Error('bad order for col ' + colIndex + ': ' + ordered);
}
exercise(4, 8);
exercise(5, 9);
exercise(6, 10);
exercise(7, 11);
console.log('outlier sort ok');
"""
    assert "outlier sort ok" in _run_node(script)


def test_sortable_tables_still_sort_contiguous_evidence_tables():
    init_sortable = _extract_js_function(JS_SCRIPTS, "initSortableTables")
    script = DOM_HARNESS + init_sortable + r"""
const headers = [new Header(1, 'Hole'), new Header(2, 'Depth')];
const rows = [
  makeRow({1: 'H2', 2: 20}),
  makeRow({1: 'H1', 2: 10}),
  makeRow({1: 'H3', 2: 30}),
];
const table = new Table('mineral-evidence-table', headers, rows);
global.document = { querySelectorAll: sel => sel === '.sortable-table' ? [table] : [] };
initSortableTables();
headers[0].click();
if (table.tbody.rows.map(row => row.children[1].dataset.sort).join(',') !== 'H1,H2,H3') {
  throw new Error('mineral table did not sort by hole');
}
console.log('contiguous sort ok');
"""
    assert "contiguous sort ok" in _run_node(script)


def test_all_issues_sort_preserves_hole_group_headers():
    init_sortable = _extract_js_function(JS_SCRIPTS, "initSortableTables")
    script = DOM_HARNESS + init_sortable + r"""
const depthHeader = new Header(1, 'Depth');
const h1 = new Row([new Cell('H1')], ['all-issues-hole-header']);
const h2 = new Row([new Cell('H2')], ['all-issues-hole-header']);
const h1a = makeRow({1: 20, 2: 'Z'});
const h1b = makeRow({1: 10, 2: 'A'});
const h2a = makeRow({1: 5, 2: 'B'});
const h2b = makeRow({1: 1, 2: 'C'});
const table = new Table('all-issues-table', [depthHeader], [h1, h1a, h1b, h2, h2a, h2b]);
global.document = { querySelectorAll: sel => sel === '.sortable-table' ? [table] : [] };
initSortableTables();
depthHeader.click();
const order = table.tbody.rows.map(row => row.classList.contains('all-issues-hole-header') ? row.children[0].textContent : row.children[1].dataset.sort).join('|');
if (order !== 'H1|10|20|H2|1|5') throw new Error(order);
console.log('all issues grouping ok');
"""
    assert "all issues grouping ok" in _run_node(script)


def test_print_report_clears_inline_display_and_restores_one_active_tab():
    print_report = _extract_js_function(JS_SCRIPTS, "printReport")
    script = print_report + r"""
class ClassList {
  constructor(initial) { this.items = new Set(initial || []); }
  contains(name) { return this.items.has(name); }
  toggle(name, force) { if (force) this.items.add(name); else this.items.delete(name); }
}
const panels = [
  { dataset: { tab: 'overview' }, style: {}, classList: new ClassList(['active']) },
  { dataset: { tab: 'outliers' }, style: {}, classList: new ClassList([]) },
];
const buttons = [
  { dataset: { tabButton: 'overview' }, classList: new ClassList(['active']) },
  { dataset: { tabButton: 'outliers' }, classList: new ClassList([]) },
];
global.document = {
  querySelectorAll: (selector) => selector === '.tab-panel' ? panels : [],
  querySelector: (selector) => selector === '.tab-button.active' ? buttons[0] : null,
};
global.window = { print: () => {} };
global.setTimeout = (fn) => fn();
function initPlotlyCharts() {}
function activateTab(tabId) {
  panels.forEach(panel => panel.classList.toggle('active', panel.dataset.tab === tabId));
  buttons.forEach(button => button.classList.toggle('active', button.dataset.tabButton === tabId));
}
printReport();
if (panels.some(panel => panel.style.display !== '')) throw new Error('inline display not cleared');
if (panels.filter(panel => panel.classList.contains('active')).length !== 1) throw new Error('wrong active panel count');
console.log('print restore ok');
"""
    assert "print restore ok" in _run_node(script)


def test_apostrophe_entities_round_trip_through_decode_and_json_parse():
    decode_fn = _extract_js_function(JS_SCRIPTS, "decodePlotlyAttr")
    script = decode_fn + r"""
const payload = '{&quot;text&quot;:&quot;Bob&#x27;s &#39;sample&#39; &amp; assay&quot;}';
const parsed = JSON.parse(decodePlotlyAttr(payload));
if (parsed.text !== "Bob's 'sample' & assay") throw new Error(parsed.text);
console.log('decode ok');
"""
    assert "decode ok" in _run_node(script)


def test_frontend_asset_grep_guards():
    assert "var(--text)" not in CSS_STYLES
    assert "tr.source-row-highlight td" in CSS_STYLES
    assert "data && (data.length || Array.isArray(data))" not in JS_SCRIPTS
    assert "children[index + 1]" not in JS_SCRIPTS
    table_html = _render_mineralisation_evidence_table(
        [
            {
                "hole_id": "H1",
                "depth_from": 0,
                "depth_to": 1,
                "validation": "Mismatch",
                "significance": "Low",
                "logged_as": "A",
                "logged_zonation": "De",
                "assay_suggests": "Mineralised",
                "geochem": {"Fe": 60.0, "SiO2": 5.0, "Al2O3": 2.0},
            }
        ],
        "logger",
    )
    assert "▼" not in table_html
