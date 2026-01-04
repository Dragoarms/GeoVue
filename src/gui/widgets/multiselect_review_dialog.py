"""Multi-select review dialog for selecting holes from shared review folders."""

import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path
from typing import Dict, List, Callable, Optional, Set
from concurrent.futures import ThreadPoolExecutor, Future
import logging
import shutil
import os

from gui.dialog_helper import DialogHelper

# from gui.gui_manager import GUIManager
from utils.json_register_manager import JSONRegisterManager
from core.file_manager import FileManager

logger = logging.getLogger(__name__)


class MultiSelectReviewDialog:
    """Dialog for selecting multiple holes from shared review folders."""

    def __init__(
        self,
        parent: tk.Tk,
        items_by_hole: Dict[str, List[Path]],
        on_confirm: Callable[[Dict[str, List[Path]]], None],
        max_selection: int = 10,
        gui_manager=None,
    ):
        """Initialize the multi-select review dialog.

        Args:
            parent: Parent window
            items_by_hole: Dictionary mapping hole_id to list of file paths
            on_confirm: Callback function when Process is clicked
            max_selection: Maximum number of holes that can be selected
        """
        self.parent = parent
        self.items_by_hole = items_by_hole
        self.on_confirm = on_confirm
        self.max_selection = max_selection

        # Track selections and availability
        self.selection_vars: Dict[str, tk.BooleanVar] = {}
        self.availability_status: Dict[str, Dict[str, any]] = {}
        self.checkbuttons: Dict[str, ttk.Checkbutton] = {}
        self.availability_labels: Dict[str, tk.Label] = {}

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.active_checks: Dict[str, Future] = {}

        # Injected GUIManager for theming
        self.gui_manager = gui_manager
        self.theme_colors = gui_manager.theme_colors if gui_manager else {}
        self.fonts = gui_manager.fonts if gui_manager else {}

        self.file_manager = FileManager()

        # Create dialog
        self._create_dialog()

    def _create_dialog(self):
        """Create the dialog window and components."""
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            DialogHelper.t("Select Holes for Review"),
            width=600,
            height=500,
        )

        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header with selection info
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        self.selection_label = tk.Label(
            header_frame,
            text=self._get_selection_text(),
            font=self.fonts["default"],
            bg=self.theme_colors["bg"],
            fg=self.theme_colors["fg"],
        )
        self.selection_label.pack(side=tk.LEFT)

        # Select/Deselect buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)

        select_all_btn = ttk.Button(
            button_frame, text=DialogHelper.t("Select All"), command=self._select_all
        )
        select_all_btn.pack(side=tk.LEFT, padx=2)

        deselect_all_btn = ttk.Button(
            button_frame,
            text=DialogHelper.t("Deselect All"),
            command=self._deselect_all,
        )
        deselect_all_btn.pack(side=tk.LEFT, padx=2)

        # Scrollable list area
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas and scrollbar
        canvas = tk.Canvas(list_frame, bg=self.theme_colors["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame inside canvas for items
        self.items_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window(
            0, 0, anchor=tk.NW, window=self.items_frame
        )

        # Build rows
        self._build_rows()

        # Configure canvas scrolling
        self.items_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Bind canvas resize
        def configure_scroll(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", configure_scroll)

        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        process_btn = ttk.Button(
            bottom_frame,
            text=DialogHelper.t("Process") + " ✓",
            command=self._on_process,
        )
        process_btn.pack(side=tk.RIGHT, padx=(5, 0))

        cancel_btn = ttk.Button(
            bottom_frame, text=DialogHelper.t("Cancel"), command=self._on_cancel
        )
        cancel_btn.pack(side=tk.RIGHT)

        # Center dialog
        DialogHelper.center_dialog(self.dialog, self.parent)

    def _build_rows(self):
        """Build the rows for each hole."""
        # Group by project for better organization
        holes_by_project: Dict[str, List[tuple[str, List[Path]]]] = {}

        for hole_id, paths in self.items_by_hole.items():
            if paths:
                # Extract project from path structure
                # Path is like: [shared]/[project]/[hole]/[files]
                project = paths[0].parent.parent.name
                if project not in holes_by_project:
                    holes_by_project[project] = []
                holes_by_project[project].append((hole_id, paths))

        row = 0
        for project, holes in sorted(holes_by_project.items()):
            # Project header
            project_label = tk.Label(
                self.items_frame,
                text=f"--- {project} ---",
                font=self.fonts["bold"],
                bg=self.theme_colors["bg"],
                fg=self.theme_colors["accent"],
            )
            project_label.grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(10, 5)
            )
            row += 1

            for hole_id, paths in sorted(holes):
                # Checkbox
                var = tk.BooleanVar(value=False)
                self.selection_vars[hole_id] = var

                cb = ttk.Checkbutton(
                    self.items_frame,
                    variable=var,
                    command=lambda h=hole_id: self._on_toggle(h),
                )
                cb.grid(row=row, column=0, sticky="w", padx=(20, 5))
                self.checkbuttons[hole_id] = cb

                # Hole label
                hole_label = tk.Label(
                    self.items_frame,
                    text=f"{hole_id} ({len(paths)} {DialogHelper.t('files')})",
                    font=self.fonts["default"],
                    bg=self.theme_colors["bg"],
                    fg=self.theme_colors["fg"],
                )
                hole_label.grid(row=row, column=1, sticky="w", padx=5)

                # Availability label
                avail_label = tk.Label(
                    self.items_frame,
                    text="",
                    font=self.fonts["small"],
                    bg=self.theme_colors["bg"],
                    fg=self.theme_colors["secondary"],
                )
                avail_label.grid(row=row, column=2, sticky="w", padx=5)
                self.availability_labels[hole_id] = avail_label

                # Initialize availability status
                self.availability_status[hole_id] = {
                    "total": len(paths),
                    "available": 0,
                    "checking": False,
                }

                row += 1

    def _get_selection_text(self) -> str:
        """Get the selection count text."""
        selected = sum(1 for var in self.selection_vars.values() if var.get())
        text = DialogHelper.t("Selected: {count} / {max}")
        return text.format(count=selected, max=self.max_selection)

    def _update_selection_label(self):
        """Update the selection count label."""
        self.selection_label.config(text=self._get_selection_text())

    def _on_toggle(self, hole_id: str):
        """Handle checkbox toggle."""
        selected_count = sum(1 for var in self.selection_vars.values() if var.get())

        if self.selection_vars[hole_id].get():
            # Check if we've reached the limit
            if selected_count > self.max_selection:
                self.selection_vars[hole_id].set(False)
                return

            # Start availability check
            self._check_availability(hole_id)
        else:
            # Cancel any ongoing check
            if hole_id in self.active_checks:
                self.active_checks[hole_id].cancel()
                del self.active_checks[hole_id]

            # Reset availability display
            self.availability_labels[hole_id].config(text="")
            self.availability_status[hole_id]["available"] = 0
            self.availability_status[hole_id]["checking"] = False

        # Update selection count
        self._update_selection_label()

        # Update checkbox states
        self._update_checkbox_states()

    def _update_checkbox_states(self):
        """Enable/disable checkboxes based on selection limit."""
        selected_count = sum(1 for var in self.selection_vars.values() if var.get())

        for hole_id, var in self.selection_vars.items():
            if not var.get() and selected_count >= self.max_selection:
                self.checkbuttons[hole_id].config(state="disabled")
            else:
                self.checkbuttons[hole_id].config(state="normal")

    def _check_availability(self, hole_id: str):
        """Check file availability in background."""
        if self.availability_status[hole_id]["checking"]:
            return

        self.availability_status[hole_id]["checking"] = True
        self.availability_status[hole_id]["available"] = 0

        # Show checking status
        self.dialog.after(0, lambda: self._update_availability_display(hole_id))

        # Submit to thread pool
        future = self.executor.submit(self._check_files_worker, hole_id)
        self.active_checks[hole_id] = future

    def _check_files_worker(self, hole_id: str) -> int:
        """Worker thread to check file availability."""
        available = 0
        paths = self.items_by_hole[hole_id]

        for path in paths:
            try:
                # Check if file exists and is readable
                if path.exists():
                    # Try to read first 1KB to ensure it's available
                    with open(path, "rb") as f:
                        f.read(1024)
                    available += 1

                    # Update UI periodically
                    self.dialog.after(
                        0,
                        lambda a=available: self._update_availability_count(hole_id, a),
                    )
            except Exception as e:
                logger.debug("File not available: %s - %s", path, e)

        return available

    def _update_availability_count(self, hole_id: str, available: int):
        """Update availability count from worker thread."""
        self.availability_status[hole_id]["available"] = available
        self._update_availability_display(hole_id)

    def _update_availability_display(self, hole_id: str):
        """Update the availability display for a hole."""
        status = self.availability_status[hole_id]

        if status["checking"]:
            if status["available"] == status["total"]:
                # All available
                text = DialogHelper.t("Available!")
                self.availability_labels[hole_id].config(
                    text=text, fg=self.theme_colors["success"]
                )
                status["checking"] = False
            else:
                # Still checking
                text = DialogHelper.t("{available} of {total} available")
                text = text.format(available=status["available"], total=status["total"])
                self.availability_labels[hole_id].config(
                    text=text, fg=self.theme_colors["warning"]
                )

    def _select_all(self):
        """Select all checkboxes up to the limit."""
        count = 0
        for hole_id, var in self.selection_vars.items():
            if count < self.max_selection:
                if not var.get():
                    var.set(True)
                    self._on_toggle(hole_id)
                    count += 1
            else:
                break

    def _deselect_all(self):
        """Deselect all checkboxes."""
        for hole_id, var in self.selection_vars.items():
            if var.get():
                var.set(False)
                self._on_toggle(hole_id)

    def _on_process(self):
        """Process selected holes."""
        # Get selected holes that are fully available
        selected = {}
        for hole_id, var in self.selection_vars.items():
            if var.get():
                status = self.availability_status[hole_id]
                if status["available"] == status["total"]:
                    selected[hole_id] = self.items_by_hole[hole_id]

        if not selected:
            # Show warning
            msg = DialogHelper.t("No fully available holes selected")
            DialogHelper.show_warning(self.dialog, msg)
            return

        # Move selected holes to local folder
        try:
            self._move_holes_to_local(selected)

            # Close dialog and call callback
            self._cleanup()
            self.dialog.destroy()
            self.on_confirm(selected)

        except Exception as e:
            logger.error("Error moving holes to local: %s", e)
            msg = DialogHelper.t("Error moving files: {error}")
            DialogHelper.show_error(self.dialog, msg.format(error=str(e)))

    def _move_holes_to_local(self, selected: Dict[str, List[Path]]):
        """Move selected holes from shared to local review folder."""
        local_review = self.file_manager.get_local_path("temp_review")

        for hole_id, paths in selected.items():
            if paths:
                # Get source folder (parent of the files)
                source_hole_folder = paths[0].parent
                project_name = source_hole_folder.parent.name

                # Create target structure
                target_project = local_review / project_name
                target_hole = target_project / hole_id

                # Move the entire hole folder
                logger.info("Moving hole %s from shared to local", hole_id)

                # Ensure target project folder exists
                target_project.mkdir(parents=True, exist_ok=True)

                # Move the folder
                shutil.move(str(source_hole_folder), str(target_hole))

    def _on_cancel(self):
        """Cancel and close dialog."""
        self._cleanup()
        self.dialog.destroy()

    def _cleanup(self):
        """Clean up resources."""
        # Cancel any active checks
        for future in self.active_checks.values():
            future.cancel()

        # Shutdown executor
        self.executor.shutdown(wait=False)

    def show(self):
        """Show the dialog."""
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.parent.wait_window(self.dialog)
