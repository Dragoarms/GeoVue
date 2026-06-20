"""Compact TFD processing entry point for GeoVue televiewer datasets."""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from gui.dialog_helper import DialogHelper
from processing.Televiewer.paths import find_tfd_files, get_televiewer_root
from processing.Televiewer.tfd_decoder import DEFAULT_PIXELS_PER_METER
from processing.Televiewer.tfd_processor import TeleviewerBatchProcessResult, TeleviewerProcessor

from .televiewer_viewer_dialog import TeleviewerViewerDialog


class TeleviewerProcessDialog:
    """Pick TFD files/folders and process them into GeoVue shared storage."""

    def __init__(
        self,
        parent,
        gui_manager,
        file_manager,
        config_manager=None,
        logger=None,
        data_manager=None,
        register_manager=None,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        self.file_manager = file_manager
        self.config_manager = config_manager
        self.logger = logger
        self.data_manager = data_manager
        self.register_manager = register_manager
        self.processor = TeleviewerProcessor(file_manager, logger=logger, data_manager=data_manager)
        self.storage_root = get_televiewer_root(file_manager, create_if_missing=True)
        self.status_var = tk.StringVar(value="Choose a folder or .tfd files to process.")
        self.storage_var = tk.StringVar(
            value=str(self.storage_root) if self.storage_root else "Televiewer Datasets folder is not configured."
        )
        self.buttons = []
        self._worker: threading.Thread | None = None
        self._create_dialog()

    def _create_dialog(self) -> None:
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            DialogHelper.t("Process Televiewer Data (.tfd)"),
            modal=False,
            topmost=False,
        )
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])
        self.dialog.geometry("760x420")

        frame = ttk.Frame(self.dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            frame,
            text=DialogHelper.t("Process Televiewer Data (.tfd)"),
            font=self.gui_manager.fonts.get("heading"),
        )
        title.pack(anchor="w", pady=(0, 8))

        ttk.Label(frame, text=DialogHelper.t("Televiewer Datasets")).pack(anchor="w")
        storage = ttk.Entry(frame, textvariable=self.storage_var, state="readonly")
        storage.pack(fill=tk.X, pady=(2, 10))

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(0, 10))

        self.folder_button = self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Choose TFD Folder"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._choose_folder,
        )
        self.folder_button.pack(side=tk.LEFT, padx=(0, 6))
        self.buttons.append(self.folder_button)

        self.files_button = self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Choose TFD Files"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._choose_files,
        )
        self.files_button.pack(side=tk.LEFT, padx=6)
        self.buttons.append(self.files_button)

        self.viewer_button = self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Open Televiewer Viewer"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._open_viewer,
        )
        self.viewer_button.pack(side=tk.LEFT, padx=6)

        close_button = self.gui_manager.create_modern_button(
            actions,
            text=DialogHelper.t("Close"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=self.dialog.destroy,
        )
        close_button.pack(side=tk.RIGHT)

        columns = ("source", "status", "message")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=9)
        for column, label, width in (
            ("source", "Source TFD", 300),
            ("status", "Status", 90),
            ("message", "Result", 320),
        ):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, textvariable=self.status_var).pack(anchor="w", pady=(8, 0))

    def _choose_folder(self) -> None:
        initial_dir = self._initial_picker_dir()
        selected = filedialog.askdirectory(
            title=DialogHelper.t("Select folder containing TFD files"),
            initialdir=str(initial_dir) if initial_dir else None,
        )
        if selected:
            self._process_sources([Path(selected)])

    def _choose_files(self) -> None:
        initial_dir = self._initial_picker_dir()
        selected = filedialog.askopenfilenames(
            title=DialogHelper.t("Select televiewer TFD files"),
            initialdir=str(initial_dir) if initial_dir else None,
            filetypes=[
                (DialogHelper.t("Televiewer TFD files"), "*.tfd"),
                (DialogHelper.t("All files"), "*.*"),
            ],
        )
        if selected:
            self._process_sources([Path(path) for path in selected])

    def _process_sources(self, sources: list[Path]) -> None:
        tfd_paths = find_tfd_files(sources)
        self._clear_results()
        if not tfd_paths:
            self.status_var.set("No .tfd files found in the selected source.")
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Process Televiewer Data (.tfd)"),
                "No .tfd files found in the selected source.",
                message_type="warning",
            )
            return

        for path in tfd_paths:
            self.tree.insert("", "end", values=(str(path), "Queued", ""))
        self.status_var.set(f"Processing {len(tfd_paths)} TFD file(s)...")
        self._set_busy(True)
        self.dialog.update_idletasks()

        self._worker = threading.Thread(
            target=self._process_worker,
            args=(tfd_paths,),
            daemon=True,
        )
        self._worker.start()

    def _process_worker(self, tfd_paths: list[Path]) -> None:
        try:
            result = self.processor.process_tfd_paths(
                tfd_paths,
                copy_raw=False,
                enrich=True,
                pixels_per_meter=DEFAULT_PIXELS_PER_METER,
            )
            self.dialog.after(0, lambda batch=result: self._finish_processing(batch, None))
        except Exception as exc:
            self.dialog.after(0, lambda error=exc: self._finish_processing(None, error))

    def _finish_processing(
        self,
        result: TeleviewerBatchProcessResult | None,
        error: Exception | None,
    ) -> None:
        self._set_busy(False)
        if error is not None:
            if self.logger:
                self.logger.error(f"Televiewer batch processing failed: {error}")
            self.status_var.set("Televiewer processing failed.")
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Televiewer Processing Error"),
                str(error),
                message_type="error",
            )
            return

        if result is None:
            return

        self._clear_results()
        for file_result in result.file_results:
            self.tree.insert(
                "",
                "end",
                values=(
                    str(file_result.plan.source_path),
                    file_result.status,
                    file_result.message,
                ),
            )

        self.status_var.set(
            f"Processed {result.total_count} TFD file(s): "
            f"{result.decoded_count} decoded, "
            f"{result.registered_count} registered without images, "
            f"{result.failed_count} failed."
        )
        if result.failed_count:
            message_type = "warning"
            message = "Some TFD files could not be processed. See the result list for details."
        elif result.decoded_count:
            message_type = "info"
            message = "Televiewer image slices and viewer data are ready."
        else:
            message_type = "warning"
            message = "No compatible OTV image strips were found. Source TFDs were registered only."

        DialogHelper.show_message(
            self.dialog,
            DialogHelper.t("Process Televiewer Data (.tfd)"),
            DialogHelper.t(message),
            message_type=message_type,
        )

    def _clear_results(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in self.buttons:
            if hasattr(button, "set_state"):
                button.set_state(state)
            else:
                button.configure(state=state)

    def _initial_picker_dir(self) -> Path | None:
        if self.storage_root:
            return self.storage_root.parent
        if self.file_manager is not None:
            for key in ("televiewer_datasets", "approved_compartments", "compartments"):
                try:
                    path = self.file_manager.get_shared_path(key, create_if_missing=False)
                except Exception:
                    path = None
                if path:
                    return Path(path)
        return None

    def _open_viewer(self) -> None:
        TeleviewerViewerDialog(
            self.dialog,
            self.gui_manager,
            self.file_manager,
            config_manager=self.config_manager,
            register_manager=self.register_manager,
        )
