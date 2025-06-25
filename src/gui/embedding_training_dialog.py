"""
GUI dialog for generating embeddings from images and tabular data.
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk
from typing import List, Tuple
import pandas as pd

from gui.dialog_helper import DialogHelper
from processing.embedding_trainer import generate_embeddings, plot_embeddings


class EmbeddingTrainingDialog:
    """Dialog allowing users to select data and generate embeddings."""

    def __init__(self, parent, gui_manager, file_manager):
        self.parent = parent
        self.gui_manager = gui_manager
        self.file_manager = file_manager
        self.image_folder = None
        self.csv_path = None
        self.df_columns: List[str] = []
        self.column_vars: List[Tuple[str, tk.BooleanVar]] = []
        self.holeid_var = tk.StringVar()
        self.depth_var = tk.StringVar()

        self._create_dialog()

    def _create_dialog(self) -> None:
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            DialogHelper.t("Embedding Tool"),
            modal=True,
            topmost=True,
        )
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

        frame = ttk.Frame(self.dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(0, 10))

        self.gui_manager.create_modern_button(
            btn_row,
            text="Select Image Folder",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._select_folder,
        ).pack(side=tk.LEFT, padx=5)

        self.gui_manager.create_modern_button(
            btn_row,
            text="Select CSV File",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._select_csv,
        ).pack(side=tk.LEFT, padx=5)

        combo_row = ttk.Frame(frame)
        combo_row.pack(fill=tk.X, pady=(0, 10))

        self.holeid_combo = ttk.Combobox(combo_row, textvariable=self.holeid_var, state="readonly")
        self.holeid_combo.pack(side=tk.LEFT, padx=5)
        self.holeid_combo.set(DialogHelper.t("Hole ID"))

        self.depth_combo = ttk.Combobox(combo_row, textvariable=self.depth_var, state="readonly")
        self.depth_combo.pack(side=tk.LEFT, padx=5)
        self.depth_combo.set(DialogHelper.t("Depth"))

        self.checkbox_frame = ttk.LabelFrame(frame, text=DialogHelper.t("Select Input Columns"))
        self.checkbox_frame.pack(fill=tk.BOTH, expand=True)

        self.gui_manager.create_modern_button(
            frame,
            text="Generate Embeddings",
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._process,
        ).pack(pady=10)

    def _select_folder(self) -> None:
        folder = filedialog.askdirectory(title=DialogHelper.t("Select Image Folder"))
        if folder:
            self.image_folder = folder

    def _select_csv(self) -> None:
        path = filedialog.askopenfilename(title=DialogHelper.t("Select CSV File"), filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        self.csv_path = path
        df = pd.read_csv(path)
        self.df_columns = df.columns.tolist()
        self.holeid_combo["values"] = self.df_columns
        self.depth_combo["values"] = self.df_columns
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        self.column_vars.clear()
        for col in self.df_columns:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.checkbox_frame, text=col, variable=var)
            chk.pack(anchor="w")
            self.column_vars.append((col, var))

    def _process(self) -> None:
        if not (self.image_folder and self.csv_path and self.holeid_var.get() and self.depth_var.get()):
            DialogHelper.show_message(self.dialog, DialogHelper.t("Error"), DialogHelper.t("Missing inputs"), message_type="error")
            return

        df = pd.read_csv(self.csv_path)
        image_files = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        images = []
        tabular = []
        for path in image_files:
            fname = os.path.basename(path)
            holeid, depth = self._extract_info(fname)
            match = df[(df[self.holeid_var.get()] == holeid) & (df[self.depth_var.get()] == depth)]
            if len(match) == 1:
                row = match.iloc[0]
                values = [float(row[col]) for col, var in self.column_vars if var.get()]
                images.append(path)
                tabular.append(values)

        if not images:
            DialogHelper.show_message(self.dialog, DialogHelper.t("Info"), DialogHelper.t("No matches found"), message_type="info")
            return

        embeddings = generate_embeddings(images, tabular)
        plot_path = os.path.join(self.file_manager.dir_structure["debugging"], "embedding_plot.png")
        plot_embeddings(embeddings, plot_path, gui_manager=self.gui_manager)
        self.file_manager.save_embedding_plot(plot_path)
        DialogHelper.show_message(self.dialog, DialogHelper.t("Success"), DialogHelper.t("Embedding plot saved"), message_type="info")

    @staticmethod
    def _extract_info(filename: str) -> Tuple[str, int]:
        import re
        match = re.search(r"(.*?)_CC_(\d{3})", filename)
        if match:
            return match.group(1), int(match.group(2))
        return "", -1

