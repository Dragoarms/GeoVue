"""
gui/snowflake_user_dialog.py

Compact dialog shown once to existing installs that have no snowflake_user
stored in their config.  Asks for the user's Fortescue email address,
optionally tests it via SSO, then saves it to ConfigManager.

Called automatically from main.py after the main window has loaded,
via root.after() — never blocks startup.

Author: George Symonds
"""

import tkinter as tk
from tkinter import ttk
import logging
import threading
import getpass

from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton

logger = logging.getLogger(__name__)


class SnowflakeUserDialog:
    """
    One-time dialog to capture the user's Snowflake login email.

    Usage:
        dlg = SnowflakeUserDialog(parent, gui_manager, config_manager)
        email = dlg.show()   # returns email string or "" if skipped
    """

    def __init__(self, parent, gui_manager, config_manager):
        self.parent = parent
        self.gui_manager = gui_manager
        self.config_manager = config_manager
        self.tc = gui_manager.theme_colors
        self.fonts = gui_manager.fonts
        self._result: str = ""
        self._test_thread: threading.Thread | None = None

    # ── public ──────────────────────────────────────────────────────────────

    def show(self) -> str:
        """
        Show the dialog modally.
        Returns the saved email, or "" if the user skipped.
        """
        self._build()
        DialogHelper.center_dialog(
            self.dialog, parent=self.parent,
            min_width=480, min_height=260,
            max_width=600, max_height=400,
        )
        self.dialog.wait_window()
        return self._result

    # ── build ────────────────────────────────────────────────────────────────

    def _build(self):
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            "Snowflake Account Setup",
            modal=True,
            topmost=True,
        )
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_skip)

        outer = tk.Frame(self.dialog, bg=self.tc["background"])
        outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        # ── header ────────────────────────────────────────────────────────
        tk.Label(
            outer,
            text="❄  Connect to Snowflake",
            font=self.fonts.get("heading", ("Arial", 13, "bold")),
            bg=self.tc["background"],
            fg=self.tc["text"],
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            outer,
            text=(
                "GeoVue can load live drillhole data from Snowflake.\n"
                "Enter your Fortescue email address to enable this."
            ),
            font=self.fonts.get("normal", ("Arial", 10)),
            bg=self.tc["background"],
            fg=self.tc["subtext"],
            justify=tk.LEFT,
            wraplength=440,
        ).pack(anchor="w", pady=(0, 14))

        # ── email row ─────────────────────────────────────────────────────
        email_row = tk.Frame(outer, bg=self.tc["background"])
        email_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            email_row,
            text="Email:",
            font=self.fonts.get("normal", ("Arial", 10)),
            bg=self.tc["background"],
            fg=self.tc["text"],
            width=8,
            anchor="e",
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._email_var = tk.StringVar(value=self._guess_email())
        self._email_entry = tk.Entry(
            email_row,
            textvariable=self._email_var,
            font=self.fonts.get("normal", ("Arial", 10)),
            bg=self.tc["field_bg"],
            fg=self.tc["text"],
            insertbackground=self.tc["text"],
            relief=tk.FLAT,
            bd=0,
            highlightbackground=self.tc["field_border"],
            highlightthickness=1,
        )
        self._email_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._email_entry.bind("<Return>", lambda e: self._on_save())

        # ── test row ──────────────────────────────────────────────────────
        test_row = tk.Frame(outer, bg=self.tc["background"])
        test_row.pack(fill=tk.X, pady=(0, 14))

        self._test_btn = ModernButton(
            test_row,
            text="Test Connection",
            color=self.tc["accent_blue"],
            command=self._on_test,
            theme_colors=self.tc,
            fonts=self.fonts,
        )
        self._test_btn.pack(side=tk.LEFT)

        self._status_label = tk.Label(
            test_row,
            text="",
            font=self.fonts.get("small", ("Arial", 9)),
            bg=self.tc["background"],
            fg=self.tc["subtext"],
            wraplength=280,
            justify=tk.LEFT,
        )
        self._status_label.pack(side=tk.LEFT, padx=(10, 0))

        # ── separator ─────────────────────────────────────────────────────
        sep = tk.Frame(outer, bg=self.tc.get("border", "#2d333d"), height=1)
        sep.pack(fill=tk.X, pady=(0, 12))

        # ── button row ────────────────────────────────────────────────────
        btn_row = tk.Frame(outer, bg=self.tc["background"])
        btn_row.pack(fill=tk.X)

        ModernButton(
            btn_row,
            text="Skip for Now",
            color=self.tc["secondary_bg"],
            command=self._on_skip,
            theme_colors=self.tc,
            fonts=self.fonts,
        ).pack(side=tk.LEFT)

        ModernButton(
            btn_row,
            text="Save & Connect",
            color=self.tc["accent_green"],
            command=self._on_save,
            theme_colors=self.tc,
            fonts=self.fonts,
        ).pack(side=tk.RIGHT)

    # ── actions ──────────────────────────────────────────────────────────────

    def _on_test(self):
        email = self._email_var.get().strip()
        if not email or "@" not in email:
            self._set_status("Please enter a valid email address.", "red")
            return

        self._test_btn.configure(state="disabled")
        self._set_status("Opening browser for SSO login…", "blue")

        def _run():
            success, msg = False, ""
            try:
                import snowflake.connector
                conn = snowflake.connector.connect(
                    account="FMG-WN74261",
                    user=email,
                    warehouse="WH_DA_EXPLORATION",
                    authenticator="externalbrowser",
                    login_timeout=60,
                )
                cur = conn.cursor()
                cur.execute("SELECT CURRENT_USER()")
                sf_user = cur.fetchone()[0]
                cur.close()
                conn.close()
                success = True
                msg = f"✓ Connected as {sf_user}"
            except ImportError:
                msg = "snowflake-connector-python not installed"
            except Exception as e:
                msg = f"✗ {str(e)[:100]}"

            def _apply():
                self._test_btn.configure(state="normal")
                color = "green" if success else "red"
                self._set_status(msg, color)

            if self.dialog.winfo_exists():
                self.dialog.after(0, _apply)

        self._test_thread = threading.Thread(
            target=_run, daemon=True, name="sf-user-test"
        )
        self._test_thread.start()

    def _on_save(self):
        email = self._email_var.get().strip()
        if email and "@" not in email:
            self._set_status("Please enter a valid email address.", "red")
            return
        self._result = email
        if email and self.config_manager:
            self.config_manager.set("snowflake_user", email)
            logger.info(f"Snowflake user saved to config: {email}")
        self.dialog.destroy()

    def _on_skip(self):
        self._result = ""
        logger.info("Snowflake user dialog skipped")
        self.dialog.destroy()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = "blue"):
        color_map = {
            "blue":  self.tc.get("accent_blue",  "#4d9be6"),
            "green": self.tc.get("accent_green", "#47b881"),
            "red":   self.tc.get("accent_red",   "#e05252"),
        }
        fg = color_map.get(color, self.tc["subtext"])
        if self.dialog.winfo_exists():
            self._status_label.config(text=text, fg=fg)

    @staticmethod
    def _guess_email() -> str:
        """Best-effort pre-fill from Windows username."""
        try:
            return f"{getpass.getuser()}@fortescue.com"
        except Exception:
            return "@fortescue.com"
