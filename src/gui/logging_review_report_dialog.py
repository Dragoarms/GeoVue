import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Any, Dict, List, Optional, Tuple

from gui.dialog_helper import DialogHelper
from gui.widgets.themed_date_entry import create_themed_date_entry
from processing.logging_review_report import (
    generate_logger_reports,
    get_logger_list_and_date_options,
    prepare_logging_review_data,
)

logger = logging.getLogger(__name__)


DEFAULT_PAGES = {
    "cover": "Cover Page",
    "summary_stats": "Summary Statistics",
    "comment_stats": "Comment Statistics",
    "fines_accuracy": "Fines Accuracy",
    "grouping_accuracy": "Grouping Accuracy",
    "outliers": "Top Outliers",
}


class LoggingReviewReportDialog:
    def __init__(
        self,
        parent: tk.Widget,
        gui_manager: Any,
        data_coordinator: Any,
        translator: Optional[Any] = None,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        self.data_coordinator = data_coordinator
        self.t = translator if translator else (lambda x: x)

        self.dialog = DialogHelper.create_dialog(
            parent, "Logging Review Report", modal=True, topmost=True
        )
        self.dialog.resizable(True, True)
        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.dialog)
            self.gui_manager.apply_theme(self.dialog)

        self.output_path_var = tk.StringVar()
        self.top_n_var = tk.IntVar(value=50)
        self.image_mode_var = tk.StringVar(value="thumbnail")

        self.date_from_var = tk.StringVar()
        self.date_to_var = tk.StringVar()

        self.page_vars: Dict[str, tk.BooleanVar] = {
            key: tk.BooleanVar(value=True) for key in DEFAULT_PAGES.keys()
        }
        self._prep_cache: Optional[Dict[str, Any]] = None
        self._prep_queue: queue.Queue = queue.Queue()
        self._logger_counts: Dict[str, Tuple[int, int, int]] = {}  # logger -> (assays, logging, holes)
        self._tooltip_window: Optional[tk.Toplevel] = None
        self._light_options: Optional[Dict[str, Any]] = None  # logger list + date options (no full prep)

        self._build_ui()
        self._load_light_options()
        self.dialog.update_idletasks()
        DialogHelper.center_dialog(
            self.dialog,
            self.parent,
            min_width=780,
            min_height=580,
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self.dialog, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        theme_colors = self.gui_manager.theme_colors if self.gui_manager else None
        fonts = self.gui_manager.fonts if self.gui_manager else None
        normal_font = ("TkDefaultFont", 10)
        if fonts:
            normal_font = fonts.get("normal") or fonts.get("entry") or normal_font

        # Progress (shown only when generating report)
        self.progress_frame = ttk.Frame(container)
        self.progress_label = ttk.Label(self.progress_frame, text=self.t("Preparing data..."))
        self.progress_label.pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, mode="indeterminate", length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=(4, 0))

        # Date range
        date_frame = ttk.LabelFrame(container, text=self.t("Date Range"))
        date_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(date_frame, text=self.t("From")).grid(row=0, column=0, padx=5, pady=5)
        if self.gui_manager and theme_colors and normal_font:
            date_from_entry = create_themed_date_entry(
                date_frame,
                self.date_from_var,
                theme_colors,
                normal_font,
                width=18,
            )
        else:
            date_from_entry = ttk.Entry(
                date_frame, textvariable=self.date_from_var, width=18
            )
        date_from_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(date_frame, text=self.t("To")).grid(row=0, column=2, padx=5, pady=5)
        if self.gui_manager and theme_colors and normal_font:
            date_to_entry = create_themed_date_entry(
                date_frame,
                self.date_to_var,
                theme_colors,
                normal_font,
                width=18,
            )
        else:
            date_to_entry = ttk.Entry(
                date_frame, textvariable=self.date_to_var, width=18
            )
        date_to_entry.grid(row=0, column=3, padx=5, pady=5)

        # Logger selection
        logger_frame = ttk.LabelFrame(container, text=self.t("Logger"))
        logger_frame.pack(fill=tk.X, pady=(0, 8))
        logger_btn_frame = ttk.Frame(logger_frame)
        logger_btn_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        if self.gui_manager and theme_colors:
            select_all_btn = self.gui_manager.create_modern_button(
                logger_btn_frame,
                text=self.t("Select all"),
                color=theme_colors["accent_blue"],
                command=self._on_select_all_loggers,
            )
        else:
            select_all_btn = ttk.Button(
                logger_btn_frame,
                text=self.t("Select all"),
                command=self._on_select_all_loggers,
            )
        select_all_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.logger_listbox = tk.Listbox(
            logger_frame, selectmode=tk.EXTENDED, height=6, exportselection=False
        )
        if theme_colors:
            self.logger_listbox.configure(
                background=theme_colors["field_bg"],
                foreground=theme_colors["text"],
                selectbackground=theme_colors.get(
                    "accent_blue", theme_colors["field_border"]
                ),
                selectforeground="#ffffff",
                highlightbackground=theme_colors["field_border"],
                highlightcolor=theme_colors["field_border"],
                relief=tk.FLAT,
                bd=1,
            )
        self.logger_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.logger_listbox.bind("<Motion>", self._on_logger_motion)
        self.logger_listbox.bind("<Leave>", self._on_logger_leave)

        # Top N
        topn_frame = ttk.LabelFrame(container, text=self.t("Outlier Count"))
        topn_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(topn_frame, text=self.t("Top N")).pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(topn_frame, from_=10, to=100, textvariable=self.top_n_var, width=8).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        # Image mode
        image_frame = ttk.LabelFrame(container, text=self.t("Images"))
        image_frame.pack(fill=tk.X, pady=(0, 8))
        for label, value in (
            ("Thumbnails", "thumbnail"),
            ("No images", "none"),
            ("Embedded originals", "embedded"),
        ):
            ttk.Radiobutton(
                image_frame,
                text=self.t(label),
                value=value,
                variable=self.image_mode_var,
            ).pack(side=tk.LEFT, padx=8, pady=5)

        # Page selection
        page_frame = ttk.LabelFrame(container, text=self.t("Report Pages"))
        page_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        row = 0
        for key, label in DEFAULT_PAGES.items():
            if self.gui_manager:
                checkbox = self.gui_manager.create_custom_checkbox(
                    page_frame, self.t(label), self.page_vars[key]
                )
                checkbox.grid(
                    row=row // 2, column=row % 2, sticky=tk.W, padx=8, pady=4
                )
            else:
                ttk.Checkbutton(
                    page_frame, text=self.t(label), variable=self.page_vars[key]
                ).grid(row=row // 2, column=row % 2, sticky=tk.W, padx=8, pady=4)
            row += 1

        # Output folder
        output_frame = ttk.LabelFrame(container, text=self.t("Output Folder"))
        output_frame.pack(fill=tk.X, pady=(0, 8))
        if self.gui_manager:
            output_entry = self.gui_manager.create_entry_with_validation(
                output_frame, self.output_path_var
            )
        else:
            output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        if self.gui_manager and theme_colors:
            browse_button = self.gui_manager.create_modern_button(
                output_frame,
                text=self.t("Browse"),
                color=theme_colors["accent_blue"],
                command=self._pick_output_folder,
            )
        else:
            browse_button = ttk.Button(
                output_frame,
                text=self.t("Browse"),
                command=self._pick_output_folder,
            )
        browse_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Action buttons
        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(8, 0))
        if self.gui_manager and theme_colors:
            generate_button = self.gui_manager.create_modern_button(
                buttons,
                text=self.t("Generate"),
                color=theme_colors["accent_green"],
                command=self._on_generate,
            )
            generate_button.pack(side=tk.RIGHT, padx=5)
            close_button = self.gui_manager.create_modern_button(
                buttons,
                text=self.t("Close"),
                color=theme_colors["accent_red"],
                command=self._on_close,
            )
            close_button.pack(side=tk.RIGHT, padx=5)
        else:
            ttk.Button(buttons, text=self.t("Generate"), command=self._on_generate).pack(
                side=tk.RIGHT, padx=5
            )
            ttk.Button(buttons, text=self.t("Close"), command=self._on_close).pack(
                side=tk.RIGHT, padx=5
            )

    def _pick_output_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self.dialog)
        if selected:
            self.output_path_var.set(selected)

    def _on_select_all_loggers(self) -> None:
        """Select all loggers in the list (generates one report per logger in chosen period)."""
        self.logger_listbox.selection_set(0, tk.END)

    def _load_light_options(self) -> None:
        """Load logger list and date range only (no heavy merge). Full prep runs on Generate."""
        if not self.data_coordinator or not self.data_coordinator.is_initialized:
            return
        self._light_options = get_logger_list_and_date_options(self.data_coordinator)
        logger_values = self._light_options.get("logger_values") or []
        for value in logger_values:
            self.logger_listbox.insert(tk.END, value)
        if logger_values:
            self.logger_listbox.selection_set(0)
        date_from = self._light_options.get("date_from") or ""
        date_to = self._light_options.get("date_to") or ""
        if date_from:
            self.date_from_var.set(date_from)
        if date_to:
            self.date_to_var.set(date_to)

    def _on_logger_motion(self, event: tk.Event) -> None:
        """Show tooltip with assays / logging / holes for hovered logger."""
        if not self._logger_counts:
            return
        idx = self.logger_listbox.nearest(event.y)
        if idx < 0 or idx >= self.logger_listbox.size():
            self._hide_tooltip()
            return
        logger_val = self.logger_listbox.get(idx)
        counts = self._logger_counts.get(logger_val)
        if not counts:
            self._hide_tooltip()
            return
        assays, logging, holes = counts
        text = self.t("Assays: {}  Logging intervals: {}  Holes: {}").format(
            assays, logging, holes
        )
        self._show_tooltip(event.x_root, event.y_root, text)

    def _on_logger_leave(self, event: tk.Event) -> None:
        self._hide_tooltip()

    def _show_tooltip(self, x: int, y: int, text: str) -> None:
        self._hide_tooltip()
        self._tooltip_window = tk.Toplevel(self.dialog)
        self._tooltip_window.wm_overrideredirect(True)
        self._tooltip_window.wm_geometry(f"+{x + 12}+{y + 12}")
        label = ttk.Label(self._tooltip_window, text=text, padding=6)
        label.pack()

    def _hide_tooltip(self) -> None:
        if self._tooltip_window:
            try:
                self._tooltip_window.destroy()
            except tk.TclError:
                pass
            self._tooltip_window = None

    def _on_generate(self) -> None:
        if not self.output_path_var.get().strip():
            DialogHelper.show_message(
                self.dialog,
                self.t("Missing Output Folder"),
                self.t("Please select an output folder before generating reports."),
                message_type="warning",
            )
            return

        selected_indices = self.logger_listbox.curselection()
        logger_values = [self.logger_listbox.get(i) for i in selected_indices]
        if not logger_values:
            DialogHelper.show_message(
                self.dialog,
                self.t("No Loggers Selected"),
                self.t("Please select at least one logger before generating reports."),
                message_type="warning",
            )
            return
        page_options = {key: var.get() for key, var in self.page_vars.items()}

        def run_prep_then_generate() -> None:
            try:
                def progress_cb(msg: str, pct: float) -> None:
                    self._prep_queue.put(("progress", msg, pct))

                prepped = prepare_logging_review_data(
                    self.data_coordinator, progress_callback=progress_cb
                )
                self._prep_queue.put(("prep_done", prepped))
            except Exception as e:
                self._prep_queue.put(("error", str(e)))

        self.progress_frame.pack(fill=tk.X, pady=(0, 8))
        self.progress_label.config(text=self.t("Preparing data..."))
        self.progress_bar.start(10)
        self._prep_cache = None
        threading.Thread(target=run_prep_then_generate, daemon=True).start()
        self._poll_then_generate(logger_values, page_options)

    def _poll_then_generate(
        self, logger_values: List[str], page_options: Dict[str, bool]
    ) -> None:
        """Poll prep queue; when done run generate_logger_reports and show result."""
        try:
            while True:
                item = self._prep_queue.get_nowait()
                if item[0] == "progress":
                    _, msg, _pct = item
                    self.progress_label.config(text=msg)
                elif item[0] == "prep_done":
                    self._prep_cache = item[1]
                    self.progress_bar.stop()
                    self.progress_frame.pack_forget()
                    self._do_generate(logger_values, page_options)
                    return
                elif item[0] == "error":
                    self.progress_bar.stop()
                    self.progress_frame.pack_forget()
                    DialogHelper.show_message(
                        self.dialog,
                        self.t("Report Error"),
                        self.t("Failed to prepare data.") + "\n" + str(item[1]),
                        message_type="error",
                    )
                    return
        except queue.Empty:
            pass
        self.dialog.after(200, lambda: self._poll_then_generate(logger_values, page_options))

    def _do_generate(
        self, logger_values: List[str], page_options: Dict[str, bool]
    ) -> None:
        """Run report generation with current prepped_data and show result."""
        try:
            image_mode = self.image_mode_var.get()
            output_files, skipped_loggers = generate_logger_reports(
                data_coordinator=self.data_coordinator,
                output_dir=self.output_path_var.get().strip(),
                date_from=self.date_from_var.get().strip() or None,
                date_to=self.date_to_var.get().strip() or None,
                logger_values=logger_values,
                top_n=int(self.top_n_var.get()),
                page_options=page_options,
                include_images=image_mode != "none",
                image_mode=image_mode,
                output_format="HTML",
                prepped_data=self._prep_cache,
            )
            if not output_files and skipped_loggers:
                DialogHelper.show_message(
                    self.dialog,
                    self.t("No Reports Generated"),
                    self.t("None of the selected loggers have data in the selected date range.")
                    + "\n\n"
                    + self.t("Skipped:")
                    + " "
                    + ", ".join(str(s) for s in skipped_loggers),
                    message_type="warning",
                )
            elif skipped_loggers:
                DialogHelper.show_message(
                    self.dialog,
                    self.t("Report Generated"),
                    self.t("Reports generated for {} logger(s).").format(len(output_files))
                    + "\n\n"
                    + self.t("The following had no data in the selected date range and were skipped:")
                    + "\n"
                    + ", ".join(str(s) for s in skipped_loggers),
                    message_type="info",
                )
            else:
                DialogHelper.show_message(
                    self.dialog,
                    self.t("Report Generated"),
                    self.t("Reports generated successfully."),
                    message_type="info",
                )
        except Exception as e:
            logger.exception("Report generation failed")
            DialogHelper.show_message(
                self.dialog,
                self.t("Report Error"),
                self.t("Failed to generate reports.") + f"\n{str(e)}",
                message_type="error",
            )

    def _on_close(self) -> None:
        self.dialog.destroy()
