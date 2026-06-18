"""
Chip Tray Inventory Dialog for GeoVue.

Displays a sortable table of ALL expected tray intervals across RC holes,
showing image counts and status (Good / Pending / Lost). Allows bulk-toggling
of Lost status and exporting filtered missing-tray PDF reports.

Status logic:
    Good    — compartment images exist for the interval (auto-derived)
    Pending — no compartments AND not marked Lost (default, implicit)
    Lost    — user-toggled; samples are permanently unavailable

Only "lost" entries are persisted in the shared JSON file.

Author: George Symonds / Claude
"""

import logging
import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton

if TYPE_CHECKING:
    from processing.DataManager.data_coordinator import DataCoordinator

logger = logging.getLogger(__name__)


# ── Data container ───────────────────────────────────────────────────

@dataclass
class TrayInterval:
    """One expected 20m tray interval with image counts."""
    project_code: str
    start_date: str
    hole_id: str
    depth_from: int
    depth_to: int
    original_count: int       # original tray images found
    compartment_count: int    # extracted compartment images found
    expected_compartments: int  # expected (typically 20)
    drilling_method: str = ""


# ── Inventory builder (runs on MissingTrayAnalyzer internals) ────────

def build_full_inventory(analyzer, collar_depths=None) -> List[TrayInterval]:
    """
    Build a flat list of ALL expected 20m intervals for RC holes,
    with original-image and compartment counts for each.

    Args:
        analyzer: Initialized MissingTrayAnalyzer instance.
        collar_depths: Optional {hole_id: (min_depth, max_depth)} from Snowflake.

    Returns:
        List[TrayInterval] sorted by project → start_date → hole → depth.
    """
    collar_meta = analyzer._build_collar_metadata()
    hole_depths = analyzer._get_hole_depths(collar_depths)

    # Filter to RC only (same logic as analyzer.run)
    rc_pattern = re.compile(r'^[A-Z]{2}\d{4}$')
    rc_depths = {}
    for h, d in hole_depths.items():
        method = collar_meta.get(h, {}).get("method", "").upper()
        if method == "RC":
            rc_depths[h] = d
        elif method in ("DIAMOND", "CHANNEL", "DD", "ADIT"):
            continue
        elif method == "" and rc_pattern.match(h):
            rc_depths[h] = d

    existing_trays = analyzer._build_existing_tray_set()
    existing_compartments = analyzer._build_existing_compartment_set()
    expected_intervals = analyzer._get_expected_intervals()

    inventory: List[TrayInterval] = []

    for hole_id, max_depth in sorted(rc_depths.items()):
        meta = collar_meta.get(hole_id, {})
        project = meta.get("project", "")
        start = meta.get("start_date", "")
        method = meta.get("method", "")
        hole_trays = existing_trays.get(hole_id, set())
        hole_comps = existing_compartments.get(hole_id, set())

        depth = 0
        while depth < max_depth:
            from_depth = depth
            to_depth = min(depth + 20, int(max_depth))

            # Count originals that overlap this interval
            orig_count = sum(
                1 for (af, at) in hole_trays
                if from_depth < at and af < to_depth
            )

            # Count compartments whose depth_to falls in [from+1 .. to]
            comp_count = sum(
                1 for d in hole_comps
                if from_depth < d <= to_depth
            )

            # Expected compartments
            hole_expected = expected_intervals.get(hole_id, set())
            expected_in_tray = {
                d for d in hole_expected
                if from_depth < d <= to_depth
            }
            if not expected_in_tray and to_depth > from_depth:
                expected_in_tray = set(range(from_depth + 1, to_depth + 1))
            exp_count = len(expected_in_tray)

            inventory.append(TrayInterval(
                project_code=project,
                start_date=start,
                hole_id=hole_id,
                depth_from=from_depth,
                depth_to=to_depth,
                original_count=orig_count,
                compartment_count=comp_count,
                expected_compartments=exp_count,
                drilling_method=method,
            ))

            depth += 20

    inventory.sort(key=lambda t: (t.project_code, t.start_date, t.hole_id, t.depth_from))
    logger.info(f"Built full inventory: {len(inventory)} intervals across {len(rc_depths)} RC holes")
    return inventory


# ── Dialog ───────────────────────────────────────────────────────────

class ChipTrayInventoryDialog:
    """
    Interactive chip tray inventory with sortable table, status toggling,
    and PDF export.
    """

    # Column definitions: (id, header, width, anchor, sort_type)
    COLUMNS = [
        ("project",   "Project",      70,  "center", "text"),
        ("start_date","Start Date",  90,  "center", "text"),
        ("hole_id",   "Hole ID",     85,  "center", "text"),
        ("from",      "From",        55,  "e",      "num"),
        ("to",        "To",          55,  "e",      "num"),
        ("originals", "Originals",   70,  "center", "num"),
        ("comps",     "Compartments",100, "center", "num"),
        ("expected",  "Expected",    70,  "center", "num"),
        ("status",    "Status",      80,  "center", "text"),
    ]

    def __init__(self, parent, data_coordinator, register_manager=None, gui_manager=None):
        """
        Args:
            parent: Parent tk window.
            data_coordinator: Initialized DataCoordinator.
            register_manager: JSONRegisterManager for persisting Lost status.
            gui_manager: Optional GUIManager for theming.
        """
        self.parent = parent
        self.data_coordinator = data_coordinator
        self.gui_manager = gui_manager
        self._register_manager = register_manager

        if gui_manager:
            self.theme_colors = gui_manager.theme_colors
            self.fonts = gui_manager.fonts
        else:
            self.theme_colors = {
                "background": "#2b2b2b", "secondary_bg": "#3c3c3c",
                "text": "#ffffff", "field_bg": "#1e1e1e",
                "accent_green": "#5aa06c", "accent_red": "#ff6b6b",
                "accent_blue": "#4a90c0",
            }
            self.fonts = {
                "normal": ("Arial", 10), "heading": ("Arial", 12, "bold"),
                "small": ("Arial", 9),
            }

        # Data
        self.inventory: List[TrayInterval] = []
        self.lost_keys: Dict[str, dict] = {}    # loaded from JSON
        self._all_items: List[dict] = []         # flat dicts for treeview
        self._items_by_key: Dict[str, dict] = {} # key → item for O(1) lookup
        self._filtered_items: List[dict] = []    # after filter

        # Sort state
        self._sort_col = "hole_id"
        self._sort_reverse = False

        # Get username for audit trail
        self._username = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))

        self._build_gui()
        self._start_analysis()

    def _build_gui(self):
        """Build the dialog window and all widgets."""
        self.dialog = DialogHelper.create_dialog(
            self.parent, "Chip Tray Inventory", modal=False, topmost=False,
        )
        self.dialog.geometry("1200x750")
        self.dialog.minsize(900, 500)

        bg = self.theme_colors["background"]

        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.dialog)

        # ── Filter bar ────────────────────────────────────────────
        filter_frame = tk.Frame(self.dialog, bg=bg)
        filter_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Project filter
        tk.Label(filter_frame, text="Project:", bg=bg, fg=self.theme_colors["text"],
                 font=self.fonts["normal"]).pack(side=tk.LEFT, padx=(0, 5))

        self.project_var = tk.StringVar(value="All")
        self.project_combo = ttk.Combobox(
            filter_frame, textvariable=self.project_var,
            state="readonly", width=12, font=self.fonts["normal"],
        )
        self.project_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.project_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Status filter
        tk.Label(filter_frame, text="Status:", bg=bg, fg=self.theme_colors["text"],
                 font=self.fonts["normal"]).pack(side=tk.LEFT, padx=(0, 5))

        self.status_var = tk.StringVar(value="All")
        self.status_combo = ttk.Combobox(
            filter_frame, textvariable=self.status_var,
            values=["All", "Good", "Pending", "Lost"],
            state="readonly", width=10, font=self.fonts["normal"],
        )
        self.status_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.status_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Hole ID search
        tk.Label(filter_frame, text="Hole ID:", bg=bg, fg=self.theme_colors["text"],
                 font=self.fonts["normal"]).pack(side=tk.LEFT, padx=(0, 5))

        self.hole_search_var = tk.StringVar()
        self.hole_search_var.trace_add("write", lambda *a: self._apply_filters())
        hole_search_entry = tk.Entry(
            filter_frame, textvariable=self.hole_search_var,
            bg=self.theme_colors.get("field_bg", "#1e1e1e"),
            fg=self.theme_colors["text"], font=self.fonts["normal"],
            insertbackground=self.theme_colors["text"], width=12,
        )
        hole_search_entry.pack(side=tk.LEFT, padx=(0, 15))

        # Row count
        self.row_count_label = tk.Label(
            filter_frame, text="", bg=bg, fg="#999999", font=self.fonts["small"],
        )
        self.row_count_label.pack(side=tk.RIGHT, padx=5)

        # ── Treeview ──────────────────────────────────────────────
        tree_frame = tk.Frame(self.dialog, bg=bg)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        col_ids = [c[0] for c in self.COLUMNS]
        self.tree = ttk.Treeview(
            tree_frame, columns=col_ids, show="headings",
            selectmode="extended",
        )

        # Configure columns and headings
        for col_id, header, width, anchor, sort_type in self.COLUMNS:
            self.tree.heading(
                col_id, text=header,
                command=lambda c=col_id: self._sort_column(c),
            )
            self.tree.column(col_id, width=width, anchor=anchor, minwidth=40)

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Row tags for colouring
        self.tree.tag_configure("good", background="#1a3a1a", foreground="#a0d8a0")
        self.tree.tag_configure("pending", background="#3a3520", foreground="#f0c860")
        self.tree.tag_configure("lost", background="#3a1a1a", foreground="#d88080")
        # Alternating shades
        self.tree.tag_configure("good_alt", background="#1e421e", foreground="#a0d8a0")
        self.tree.tag_configure("pending_alt", background="#3e3922", foreground="#f0c860")
        self.tree.tag_configure("lost_alt", background="#3e1e1e", foreground="#d88080")

        # ── Action buttons ────────────────────────────────────────
        btn_frame = tk.Frame(self.dialog, bg=bg)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ModernButton(
            btn_frame, text="Mark Lost",
            color=self.theme_colors["accent_red"],
            command=self._mark_lost,
            theme_colors=self.theme_colors, fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            btn_frame, text="Clear Status",
            color=self.theme_colors["accent_blue"],
            command=self._clear_status,
            theme_colors=self.theme_colors, fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)

        # Spacer
        tk.Frame(btn_frame, bg=bg, width=40).pack(side=tk.LEFT)

        ModernButton(
            btn_frame, text="Select All Pending",
            color="#6a6a3a",
            command=self._select_all_pending,
            theme_colors=self.theme_colors, fonts=self.fonts,
        ).pack(side=tk.LEFT, padx=5)

        # Right side — export
        ModernButton(
            btn_frame, text="Export PDF (Pending only)",
            color=self.theme_colors["accent_green"],
            command=self._export_pdf,
            theme_colors=self.theme_colors, fonts=self.fonts,
        ).pack(side=tk.RIGHT, padx=5)

        ModernButton(
            btn_frame, text="Export CSV",
            color=self.theme_colors["accent_blue"],
            command=self._export_csv,
            theme_colors=self.theme_colors, fonts=self.fonts,
        ).pack(side=tk.RIGHT, padx=5)

        # ── Summary bar ───────────────────────────────────────────
        summary_frame = tk.Frame(self.dialog, bg=self.theme_colors.get("secondary_bg", "#3c3c3c"))
        summary_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.summary_label = tk.Label(
            summary_frame, text="Loading...", bg=self.theme_colors.get("secondary_bg", "#3c3c3c"),
            fg=self.theme_colors["text"], font=self.fonts["normal"],
            anchor="w", padx=10, pady=6,
        )
        self.summary_label.pack(fill=tk.X)

    # ── Analysis (background thread) ─────────────────────────────

    def _start_analysis(self):
        """Run analysis on a background thread."""
        self.summary_label.config(text="Running inventory analysis...")

        def _run():
            try:
                from processing.find_missing_trays import MissingTrayAnalyzer, MissingTrayStatus

                # Get collar depths
                collar_depths = None
                try:
                    from processing.DataManager.snowflake_session import get_session_manager
                    sm = get_session_manager()
                    if sm.collar_depth_ranges:
                        collar_depths = sm.collar_depth_ranges
                except Exception:
                    pass

                analyzer = MissingTrayAnalyzer(self.data_coordinator)

                inventory = build_full_inventory(analyzer, collar_depths=collar_depths)

                # Load lost entries from JSON
                lost_keys = {}
                if self._register_manager:
                    raw = self._register_manager.read_missing_tray_status()
                    # Filter to only "lost" status entries (ignore legacy statuses)
                    lost_keys = {
                        k: v for k, v in raw.items()
                        if v.get("status") == "lost"
                    }

                self.dialog.after(0, lambda: self._on_analysis_complete(inventory, lost_keys))

            except Exception as e:
                logger.error(f"Inventory analysis failed: {e}", exc_info=True)
                self.dialog.after(0, lambda: self.summary_label.config(
                    text=f"Analysis failed: {e}"
                ))

        threading.Thread(target=_run, daemon=True, name="inventory-analysis").start()

    def _on_analysis_complete(self, inventory: List[TrayInterval], lost_keys: Dict[str, dict]):
        """Called on main thread when analysis finishes."""
        self.inventory = inventory
        self.lost_keys = lost_keys

        # Build flat item list
        self._all_items = []
        self._items_by_key = {}
        for t in inventory:
            key = f"{t.hole_id.upper()}_{t.depth_from}-{t.depth_to}"
            if t.compartment_count > 0:
                status = "Good"
            elif key in self.lost_keys:
                status = "Lost"
            else:
                status = "Pending"

            item_dict = {
                "key": key,
                "project": t.project_code,
                "start_date": t.start_date,
                "hole_id": t.hole_id,
                "from": t.depth_from,
                "to": t.depth_to,
                "originals": t.original_count,
                "comps": t.compartment_count,
                "expected": t.expected_compartments,
                "status": status,
            }
            self._all_items.append(item_dict)
            self._items_by_key[key] = item_dict

        # Populate project filter
        projects = sorted({item["project"] for item in self._all_items if item["project"]})
        self.project_combo["values"] = ["All"] + projects

        self._apply_filters()

    # ── Filtering & display ──────────────────────────────────────

    def _apply_filters(self):
        """Filter items and repopulate treeview."""
        proj = self.project_var.get()
        stat = self.status_var.get()
        hole_search = self.hole_search_var.get().strip().upper()

        self._filtered_items = []
        for item in self._all_items:
            if proj != "All" and item["project"] != proj:
                continue
            if stat != "All" and item["status"] != stat:
                continue
            if hole_search and hole_search not in item["hole_id"].upper():
                continue
            self._filtered_items.append(item)

        self._populate_table()
        self._update_summary()

    def _populate_table(self):
        """Clear and repopulate the treeview from _filtered_items."""
        self.tree.delete(*self.tree.get_children())

        # Sort
        sort_type = "text"
        for col_id, _, _, _, st in self.COLUMNS:
            if col_id == self._sort_col:
                sort_type = st
                break

        def sort_key(item):
            val = item.get(self._sort_col, "")
            if sort_type == "num":
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0
            return str(val).lower()

        sorted_items = sorted(self._filtered_items, key=sort_key, reverse=self._sort_reverse)

        # Insert rows
        prev_hole = None
        alt = False
        for item in sorted_items:
            # Alternate shading per hole
            if item["hole_id"] != prev_hole:
                alt = not alt
                prev_hole = item["hole_id"]

            status = item["status"]
            tag = status.lower() + ("_alt" if alt else "")

            values = (
                item["project"],
                item["start_date"],
                item["hole_id"],
                item["from"],
                item["to"],
                item["originals"],
                item["comps"],
                item["expected"],
                item["status"],
            )
            self.tree.insert("", "end", iid=item["key"], values=values, tags=(tag,))

        self.row_count_label.config(
            text=f"{len(sorted_items):,} / {len(self._all_items):,} intervals"
        )

    def _sort_column(self, col_id):
        """Sort by the clicked column header."""
        if self._sort_col == col_id:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col_id
            self._sort_reverse = False

        # Update header arrows
        for cid, header, _, _, _ in self.COLUMNS:
            arrow = ""
            if cid == col_id:
                arrow = " ▼" if self._sort_reverse else " ▲"
            self.tree.heading(cid, text=header + arrow)

        self._populate_table()

    # ── Status toggling ──────────────────────────────────────────

    def _mark_lost(self):
        """Mark all selected Pending rows as Lost."""
        selection = self.tree.selection()
        if not selection:
            DialogHelper.show_message(
                self.dialog, "No Selection",
                "Select one or more Pending rows to mark as Lost.",
                "info",
            )
            return

        changed = 0
        for iid in selection:
            item = self._find_item_by_key(iid)
            if item and item["status"] == "Pending":
                item["status"] = "Lost"
                self.lost_keys[iid] = {
                    "status": "lost",
                    "by": self._username,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                }
                changed += 1

        if changed > 0:
            self._save_lost_entries()
            self._populate_table()
            self._update_summary()
            logger.info(f"Marked {changed} intervals as Lost")

    def _clear_status(self):
        """Clear Lost status from selected rows (back to Pending)."""
        selection = self.tree.selection()
        if not selection:
            DialogHelper.show_message(
                self.dialog, "No Selection",
                "Select one or more Lost rows to clear.",
                "info",
            )
            return

        changed = 0
        for iid in selection:
            item = self._find_item_by_key(iid)
            if item and item["status"] == "Lost":
                item["status"] = "Pending"
                self.lost_keys.pop(iid, None)
                changed += 1

        if changed > 0:
            self._save_lost_entries()
            self._populate_table()
            self._update_summary()
            logger.info(f"Cleared {changed} Lost entries back to Pending")

    def _select_all_pending(self):
        """Select all visible Pending rows in the treeview."""
        pending_iids = []
        for iid in self.tree.get_children():
            vals = self.tree.item(iid, "values")
            if vals and vals[8] == "Pending":  # status column index 8
                pending_iids.append(iid)

        if pending_iids:
            self.tree.selection_set(pending_iids)
            # Scroll to first
            self.tree.see(pending_iids[0])
            self.row_count_label.config(
                text=f"{len(pending_iids)} Pending selected / {len(self._filtered_items):,} shown"
            )

    def _find_item_by_key(self, key: str) -> Optional[dict]:
        """Find an item dict by its key (O(1) lookup)."""
        return self._items_by_key.get(key)

    def _save_lost_entries(self):
        """Persist lost entries to the JSON register."""
        if not self._register_manager:
            logger.warning("No register manager — cannot save lost entries")
            return

        # Read existing data (may have other legacy statuses)
        existing = self._register_manager.read_missing_tray_status()

        # Remove all "lost" entries, then re-add current ones
        cleaned = {k: v for k, v in existing.items() if v.get("status") != "lost"}
        cleaned.update(self.lost_keys)

        self._register_manager.write_missing_tray_status(cleaned)

    # ── Summary ──────────────────────────────────────────────────

    def _update_summary(self):
        """Update the summary statistics bar."""
        total = len(self._all_items)
        good = sum(1 for i in self._all_items if i["status"] == "Good")
        pending = sum(1 for i in self._all_items if i["status"] == "Pending")
        lost = sum(1 for i in self._all_items if i["status"] == "Lost")

        good_pct = (good / total * 100) if total else 0
        accounted = good + lost
        accounted_pct = (accounted / total * 100) if total else 0

        parts = [
            f"Total: {total:,}",
            f"Good: {good:,} ({good_pct:.1f}%)",
            f"Pending: {pending:,}",
            f"Lost: {lost:,}",
            f"Accounted for: {accounted:,} ({accounted_pct:.1f}%)",
        ]
        self.summary_label.config(text="    |    ".join(parts))

    # ── Export ────────────────────────────────────────────────────

    def _export_pdf(self):
        """Export pending (unresolved) trays to PDF."""
        pending_items = [i for i in self._all_items if i["status"] == "Pending"]
        if not pending_items:
            DialogHelper.show_message(
                self.dialog, "Nothing to Export",
                "No Pending trays to export. All intervals are either Good or Lost.",
                "info",
            )
            return

        output_path = filedialog.asksaveasfilename(
            parent=self.dialog,
            title="Export Pending Trays Report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"pending_trays_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        )
        if not output_path:
            return

        try:
            from processing.find_missing_trays import (
                AnalysisResults, MissingTray, export_missing_trays_pdf,
            )

            # Build an AnalysisResults from the pending items
            results = AnalysisResults()
            results.total_holes_analysed = len({i["hole_id"] for i in self._all_items})
            results.total_expected_trays = len(self._all_items)
            results.total_existing_trays = sum(1 for i in self._all_items if i["status"] == "Good")

            for item in pending_items:
                results.missing_trays.append(MissingTray(
                    hole_id=item["hole_id"],
                    depth_from=item["from"],
                    depth_to=item["to"],
                    max_depth=item["to"],
                    project_code=item["project"],
                    start_date=item["start_date"],
                ))

            success = export_missing_trays_pdf(results, output_path)

            if success:
                DialogHelper.show_message(
                    self.dialog, "Export Complete",
                    f"Exported {len(pending_items):,} pending tray intervals.\n\n{output_path}",
                    "info",
                )
                try:
                    import subprocess
                    subprocess.Popen(["start", "", output_path], shell=True)
                except Exception:
                    pass
            else:
                DialogHelper.show_message(
                    self.dialog, "Export Failed",
                    "PDF export failed. Check the log for details.",
                    "error",
                )

        except Exception as e:
            logger.error(f"PDF export failed: {e}", exc_info=True)
            DialogHelper.show_message(
                self.dialog, "Export Error", f"Failed to export PDF:\n{e}", "error",
            )

    def _export_csv(self):
        """Export the current filtered view to CSV."""
        if not self._filtered_items:
            DialogHelper.show_message(
                self.dialog, "Nothing to Export",
                "No data to export with current filters.",
                "info",
            )
            return

        output_path = filedialog.asksaveasfilename(
            parent=self.dialog,
            title="Export Inventory CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"chip_tray_inventory_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        )
        if not output_path:
            return

        try:
            import csv
            fieldnames = ["project", "start_date", "hole_id", "from", "to",
                          "originals", "comps", "expected", "status"]
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._filtered_items)

            DialogHelper.show_message(
                self.dialog, "Export Complete",
                f"Exported {len(self._filtered_items):,} rows.\n\n{output_path}",
                "info",
            )
        except Exception as e:
            logger.error(f"CSV export failed: {e}", exc_info=True)
            DialogHelper.show_message(
                self.dialog, "Export Error", f"Failed to export CSV:\n{e}", "error",
            )
