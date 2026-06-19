"""Tk dialog for building and training lithology ML datasets."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from gui.widgets.modern_button import ModernButton
from ml_pipeline.lithology_manifest import (
    build_manifest_rows_from_images,
    summarise_rows,
    write_manifest_csv,
    write_summary_json,
)


class LithologyMLTrainingDialog:
    """Build a register-derived manifest and launch lithology training."""

    DEFAULT_TARGETS = ("BIFf", "BIFf-s", "BIFhm")

    def __init__(
        self,
        parent,
        gui_manager,
        images,
        item_manager,
        consensus_fn=None,
        metadata_fn=None,
        logger: logging.Logger | None = None,
    ):
        self.parent = parent
        self.gui_manager = gui_manager
        self.images = list(images or [])
        self.item_manager = item_manager
        self.consensus_fn = consensus_fn
        self.metadata_fn = metadata_fn
        self.logger = logger or logging.getLogger(__name__)

        self.repo_root = Path(__file__).resolve().parents[2]
        self.output_dir = self.repo_root / "ml_output" / "lithology_classifier"
        self.manifest_path: Path | None = None
        self.rows = []
        self.summary = {}
        self.process: subprocess.Popen | None = None
        self.reader_thread: threading.Thread | None = None

        self.dialog = None
        self.class_vars: dict[str, tk.BooleanVar] = {}
        self.use_consensus_var = tk.BooleanVar(value=True)
        self.max_per_class_var = tk.IntVar(value=0)
        self.epochs_var = tk.IntVar(value=30)
        self.batch_var = tk.IntVar(value=32)
        self.lr_var = tk.DoubleVar(value=0.0005)
        self.model_var = tk.StringVar(value="mobilenet_v3_small")
        self.output_dir_var = tk.StringVar(value=str(self.output_dir))

    def show(self) -> None:
        self._create_dialog()

    def _create_dialog(self) -> None:
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Lithology ML Classifier")
        self.dialog.geometry("980x720")
        self.dialog.transient(self.parent)

        if self.gui_manager:
            self.gui_manager.configure_ttk_styles(self.dialog)
            self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

        root = ttk.Frame(self.dialog, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1)

        self._create_target_section(root)
        self._create_options_section(root)
        self._create_action_section(root)
        self._create_log_section(root)

        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)
        self._append_log(f"Loaded {len(self.images):,} image records from the lithology dialog.")

    def _create_target_section(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Training Items", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        labels = [item.label for item in self.item_manager.get_active_classifications()]
        selected_defaults = {label.lower() for label in self.DEFAULT_TARGETS}
        for label in labels:
            var = tk.BooleanVar(value=label.lower() in selected_defaults)
            self.class_vars[label] = var
            ttk.Checkbutton(frame, text=label, variable=var).pack(side=tk.LEFT, padx=(0, 12))

        if not any(var.get() for var in self.class_vars.values()):
            for label in labels[:3]:
                self.class_vars[label].set(True)

    def _create_options_section(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Dataset And Training Options", padding=8)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        frame.grid_columnconfigure(1, weight=1)

        ttk.Checkbutton(
            frame,
            text="Use consensus classifications and aggregate tags from peer reviews",
            variable=self.use_consensus_var,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        ttk.Label(frame, text="Max samples per class").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0, to=100000, increment=500, textvariable=self.max_per_class_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(8, 20)
        )
        ttk.Label(frame, text="0 = use all").grid(row=1, column=2, sticky="w")

        ttk.Label(frame, text="Model").grid(row=2, column=0, sticky="w", pady=(8, 0))
        model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_var,
            values=("mobilenet_v3_small", "resnet18", "resnet34", "efficientnet_b0"),
            state="readonly",
            width=22,
        )
        model_combo.grid(row=2, column=1, sticky="w", padx=(8, 20), pady=(8, 0))

        ttk.Label(frame, text="Epochs").grid(row=2, column=2, sticky="e", pady=(8, 0))
        ttk.Spinbox(frame, from_=1, to=300, textvariable=self.epochs_var, width=8).grid(
            row=2, column=3, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(frame, text="Batch").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(frame, from_=4, to=256, increment=4, textvariable=self.batch_var, width=8).grid(
            row=3, column=1, sticky="w", padx=(8, 20), pady=(8, 0)
        )

        ttk.Label(frame, text="Learning rate").grid(row=3, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.lr_var, width=10).grid(
            row=3, column=3, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(frame, text="Output folder").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=self.output_dir_var).grid(
            row=4, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0)
        )
        ttk.Button(frame, text="Browse", command=self._choose_output_dir).grid(row=4, column=3, sticky="w", pady=(8, 0))

    def _create_action_section(self, parent) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        colors = self.gui_manager.theme_colors if self.gui_manager else {}
        ModernButton(
            frame,
            text="Preview Dataset",
            command=self.preview_dataset,
            color=colors.get("accent_blue", "#1f77b4"),
            theme_colors=colors,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ModernButton(
            frame,
            text="Write Manifest",
            command=self.write_manifest,
            color=colors.get("secondary_bg", "#6c757d"),
            theme_colors=colors,
        ).pack(side=tk.LEFT, padx=6)
        self.train_button = ModernButton(
            frame,
            text="Start Training",
            command=self.start_training,
            color=colors.get("accent_green", "#28a745"),
            theme_colors=colors,
        )
        self.train_button.pack(side=tk.LEFT, padx=6)
        self.stop_button = ModernButton(
            frame,
            text="Stop",
            command=self.stop_training,
            color="#dc3545",
            theme_colors=colors,
            enabled=False,
        )
        self.stop_button.pack(side=tk.LEFT, padx=6)

    def _create_log_section(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Training Log", padding=8)
        frame.grid(row=3, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.log_text = tk.Text(frame, wrap=tk.WORD, height=18)
        scrollbar = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _selected_labels(self) -> list[str]:
        return [label for label, var in self.class_vars.items() if var.get()]

    def _choose_output_dir(self) -> None:
        chosen = filedialog.askdirectory(
            parent=self.dialog,
            title="Select lithology classifier output folder",
            initialdir=self.output_dir_var.get(),
        )
        if chosen:
            self.output_dir_var.set(chosen)

    def _build_rows(self) -> list:
        labels = self._selected_labels()
        if len(labels) < 2:
            raise ValueError("Select at least two classes before training.")
        max_per_class = self.max_per_class_var.get()
        rows = build_manifest_rows_from_images(
            self.images,
            labels,
            consensus_fn=self.consensus_fn,
            metadata_fn=self.metadata_fn,
            use_consensus=self.use_consensus_var.get(),
            max_per_class=max_per_class if max_per_class > 0 else None,
        )
        if not rows:
            raise ValueError("No matching classified images were found for the selected classes.")
        return rows

    def preview_dataset(self) -> None:
        try:
            self.rows = self._build_rows()
            self.summary = summarise_rows(self.rows)
            self._append_log("Dataset preview:")
            self._append_log(json.dumps(self.summary, indent=2))
        except Exception as exc:
            self._append_log(f"Preview failed: {exc}")
            messagebox.showerror("Dataset Preview Failed", str(exc), parent=self.dialog)

    def write_manifest(self) -> Path | None:
        try:
            if not self.rows:
                self.rows = self._build_rows()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.output_dir_var.get())
            manifest = output_dir / "datasets" / f"lithology_training_manifest_{timestamp}.csv"
            summary = output_dir / "datasets" / f"lithology_training_summary_{timestamp}.json"
            write_manifest_csv(self.rows, manifest)
            write_summary_json(self.rows, summary)
            self.manifest_path = manifest
            self.summary = summarise_rows(self.rows)
            self._append_log(f"Manifest written: {manifest}")
            self._append_log(f"Summary written: {summary}")
            return manifest
        except Exception as exc:
            self._append_log(f"Manifest write failed: {exc}")
            messagebox.showerror("Manifest Write Failed", str(exc), parent=self.dialog)
            return None

    def start_training(self) -> None:
        if self.process and self.process.poll() is None:
            return
        manifest = self.manifest_path or self.write_manifest()
        if not manifest:
            return

        output_dir = Path(self.output_dir_var.get())
        class_arg = ",".join(self._selected_labels())
        command = [
            sys.executable,
            "-m",
            "ml_pipeline.train_lithology",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--classes",
            class_arg,
            "--model",
            self.model_var.get(),
            "--epochs",
            str(self.epochs_var.get()),
            "--batch-size",
            str(self.batch_var.get()),
            "--lr",
            str(self.lr_var.get()),
            "--no-viz",
        ]

        self._append_log("Starting training command:")
        self._append_log(" ".join(f'"{part}"' if " " in part else part for part in command))
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._append_log(f"Training launch failed: {exc}")
            messagebox.showerror("Training Launch Failed", str(exc), parent=self.dialog)
            return

        self.train_button.set_state("disabled")
        self.stop_button.set_state("normal")
        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _read_process_output(self) -> None:
        assert self.process is not None
        if self.process.stdout is not None:
            for line in self.process.stdout:
                self._append_log_threadsafe(line.rstrip())
        code = self.process.wait()
        self._append_log_threadsafe(f"Training process exited with code {code}.")
        self.dialog.after(0, self._training_finished)

    def _training_finished(self) -> None:
        self.train_button.set_state("normal")
        self.stop_button.set_state("disabled")
        best = Path(self.output_dir_var.get()) / "checkpoints" / "best_model.pt"
        if best.exists():
            self._append_log(f"Best model ready: {best}")

    def stop_training(self) -> None:
        if self.process and self.process.poll() is None:
            self._append_log("Stopping training process...")
            self.process.terminate()

    def _append_log_threadsafe(self, message: str) -> None:
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.after(0, lambda: self._append_log(message))

    def _append_log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.insert(tk.END, message + os.linesep)
        self.log_text.see(tk.END)

    def _on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno(
                "Training In Progress",
                "Training is still running. Stop it and close this dialog?",
                parent=self.dialog,
            ):
                return
            self.stop_training()
        self.dialog.destroy()
