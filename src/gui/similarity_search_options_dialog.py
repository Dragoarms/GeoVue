"""Compact options dialog for Logging Review similarity search.

The logging-review side panel is already dense, so this dialog keeps advanced
visual/chemical/spatial controls out of the main layout while still making the
search assumptions visible before a run.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Optional


_MODE_LABELS: tuple[tuple[str, str], ...] = (
    ("Visual only", "visual"),
    ("Hybrid: visual + chemical + spatial", "hybrid"),
    ("Chemical only", "chemical"),
    ("Spatial only", "spatial"),
    ("Continuity: local visual + chemical trend", "continuity"),
)

_DEFAULT_CHEMISTRY = "fe_pct, sio2_pct, al2o3_pct, p_pct, s_pct, loi_pct, mn_pct, cao_pct, mgo_pct, tio2_pct"


class SimilaritySearchOptionsDialog:
    """Modal form that returns a similarity search configuration dictionary."""

    def __init__(self, parent: tk.Misc, *, has_selected_image: bool = True) -> None:
        self.parent = parent
        self.has_selected_image = has_selected_image
        self.result: Optional[dict[str, Any]] = None

        self.window = tk.Toplevel(parent)
        self.window.title("Similarity Search")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.resizable(False, False)

        self.mode_var = tk.StringVar(value=_MODE_LABELS[0][0])
        self.use_xyz_var = tk.BooleanVar(value=True)
        self.show_all_var = tk.BooleanVar(value=True)
        self.spatial_range_var = tk.StringVar(value="50")
        self.depth_range_var = tk.StringVar(value="10")
        self.continuity_window_var = tk.StringVar(value="3")
        self.top_k_var = tk.StringVar(value="500")
        self.chemistry_var = tk.StringVar(value=_DEFAULT_CHEMISTRY)

        self._build()
        self._center()

    def show(self) -> Optional[dict[str, Any]]:
        self.parent.wait_window(self.window)
        return self.result

    def _build(self) -> None:
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        mode_frame = ttk.LabelFrame(outer, text="Search Mode", padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        mode_menu = ttk.OptionMenu(
            mode_frame,
            self.mode_var,
            self.mode_var.get(),
            *[label for label, _mode in _MODE_LABELS],
        )
        mode_menu.pack(fill=tk.X)

        chemistry_frame = ttk.LabelFrame(outer, text="Chemistry Elements", padding=8)
        chemistry_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(chemistry_frame, textvariable=self.chemistry_var, width=64).pack(fill=tk.X)

        spatial_frame = ttk.LabelFrame(outer, text="Spatial / Continuity", padding=8)
        spatial_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(
            spatial_frame,
            text="Use collar/survey 3D coordinates when available",
            variable=self.use_xyz_var,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

        ttk.Label(spatial_frame, text="3D range m").grid(row=1, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(spatial_frame, textvariable=self.spatial_range_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(spatial_frame, text="Depth range m").grid(row=1, column=2, sticky="w", padx=(12, 4))
        ttk.Entry(spatial_frame, textvariable=self.depth_range_var, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(spatial_frame, text="Continuity window m").grid(row=2, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        ttk.Entry(spatial_frame, textvariable=self.continuity_window_var, width=8).grid(row=2, column=1, sticky="w", pady=(4, 0))

        output_frame = ttk.LabelFrame(outer, text="Output", padding=8)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            output_frame,
            text="Show all ranked matches after active filters are applied",
            variable=self.show_all_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(output_frame, text="Limit when unchecked").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(output_frame, textvariable=self.top_k_var, width=8).grid(row=1, column=1, sticky="w", pady=(4, 0))

        if not self.has_selected_image:
            note = ttk.Label(
                outer,
                text="No single grid image is selected. After OK, click one image in the review grid to use as the query.",
                wraplength=430,
            )
            note.pack(fill=tk.X, pady=(0, 8))

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="OK", command=self._accept).pack(side=tk.RIGHT)

        self.window.bind("<Escape>", lambda _event: self._cancel())
        self.window.bind("<Return>", lambda _event: self._accept())

    def _center(self) -> None:
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_w = self.parent.winfo_width()
        parent_h = self.parent.winfo_height()
        x = parent_x + max(0, (parent_w - width) // 2)
        y = parent_y + max(0, (parent_h - height) // 2)
        self.window.geometry(f"+{x}+{y}")

    def _mode_value(self) -> str:
        selected = self.mode_var.get()
        for label, value in _MODE_LABELS:
            if label == selected:
                return value
        return "hybrid"

    def _float_value(self, value: str, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _accept(self) -> None:
        chemistry_columns = tuple(
            part.strip()
            for part in self.chemistry_var.get().split(",")
            if part.strip()
        )
        top_k = None
        if not self.show_all_var.get():
            try:
                top_k = max(1, int(float(self.top_k_var.get())))
            except (TypeError, ValueError):
                top_k = 500

        self.result = {
            "mode": self._mode_value(),
            "chemistry_columns": chemistry_columns,
            "use_xyz": bool(self.use_xyz_var.get()),
            "spatial_range_m": self._float_value(self.spatial_range_var.get(), 50.0),
            "depth_range_m": self._float_value(self.depth_range_var.get(), 10.0),
            "continuity_window_m": self._float_value(self.continuity_window_var.get(), 3.0),
            "top_k": top_k,
        }
        self.window.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.window.destroy()
