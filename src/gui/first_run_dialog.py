# gui/first_run_dialog.py

import os
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
import logging
from gui.dialog_helper import DialogHelper
from utils.json_register_manager import JSONRegisterManager
from gui.widgets.modern_notebook import ModernNotebook


class FirstRunDialog:
    """Dialog for first-time setup of the application."""

    def __init__(self, parent, gui_manager):
        """Initialize first run setup dialog."""
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.gui_manager = gui_manager
        self.result = None
        self.local_path = None
        self.backup_path = None

        # Get folder structure from FileManager if available
        if hasattr(gui_manager, "file_manager") and gui_manager.file_manager:
            # Use FileManager's folder names if available
            self.REQUIRED_FOLDERS = {
                "register": "Chip Tray Register",
                "images": "Images to Process",
                "processed": "Processed Original Images",
                "compartments": "Extracted Compartment Images",
                "traces": "Drillhole Traces",
                "datasets": "Drillhole Datasets",
                "cross_sections": "Cross Sections",
            }
            self.REQUIRED_SUBFOLDERS = {
                "processed": ["Approved Originals", "Rejected Originals"],
                "register": ["Register Data (Do not edit)"],
                "compartments": [
                    "Approved Compartment Images",
                    "Compartment Images for Review",
                ],
            }
        else:
            # Fallback definitions
            self.REQUIRED_FOLDERS = {
                "register": "Chip Tray Register",
                "images": "Images to Process",
                "processed": "Processed Original Images",
                "compartments": "Extracted Compartment Images",
                "traces": "Drillhole Traces",
                "datasets": "Drillhole Datasets",
                "cross_sections": "Cross Sections",
            }
            self.REQUIRED_SUBFOLDERS = {
                "processed": ["Approved Originals", "Rejected Originals"],
                "register": ["Register Data (Do not edit)"],
                "compartments": [
                    "Approved Compartment Images",
                    "Compartment Images for Review",
                ],
            }

    def show(self) -> dict:
        """Show the first run dialog and return configuration."""
        self.logger.debug(f"Parent window: {self.parent}")
        self.logger.debug(
            f"Parent exists: {self.parent.winfo_exists() if self.parent else 'No parent'}"
        )

        if self.parent:
            self.logger.debug("Withdrawing parent window")
            self.parent.withdraw()

        self.logger.debug(f"After withdrawing Parent window: {self.parent}")
        self.logger.debug(
            f"After withdrawing - Parent exists: {self.parent.winfo_exists() if self.parent else 'No parent'}"
        )

        if hasattr(self.parent, "_first_run_dialog"):
            self.logger.debug("Warning: Previous first run dialog still exists")
        self.parent._first_run_dialog = self

        # Initialize variables early (normally set in pages)
        self.local_path_var = tk.StringVar(value="C:\\GeoVue Chip Tray Photos")
        self.local_path = "C:\\GeoVue Chip Tray Photos"
        self.backup_path_var = tk.StringVar()
        self.use_backup_var = tk.BooleanVar(value=True)  # Always true, backup is required
        
        # Automatically check and create local folder structure  
        default_path = Path(self.local_path_var.get())
        if not default_path.exists():
            default_path.mkdir(parents=True, exist_ok=True)

        self.dialog = DialogHelper.create_dialog(
            self.parent, DialogHelper.t("Welcome to GeoVue"), modal=False, topmost=True
        )
        self.logger.debug(f"Dialog created: {self.dialog}")

        # Ensure dialog appears above all windows including splash
        self.dialog.lift()
        self.dialog.attributes("-topmost", True)
        self.dialog.update_idletasks()

        # Configure dialog background
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])

        # Set icon if available
        try:
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "resources", "logo.ico"
            )
            if os.path.exists(icon_path):
                self.dialog.iconbitmap(icon_path)
        except Exception as e:
            self.logger.debug(f"Could not set icon: {e}")

        # Apply theme
        if self.gui_manager:
            self.gui_manager.apply_theme(self.dialog)

        try:
            # Create main_container using grid
            main_container = ttk.Frame(self.dialog, padding="10")
            main_container.grid(row=0, column=0, sticky="nsew")

            # Configure dialog grid weights
            self.dialog.grid_rowconfigure(0, weight=1)
            self.dialog.grid_columnconfigure(0, weight=1)

            # Configure main container grid weights
            main_container.grid_rowconfigure(0, weight=0)  # Header - fixed height
            main_container.grid_rowconfigure(1, weight=1)  # Notebook - expandable
            main_container.grid_rowconfigure(2, weight=0)  # Buttons - fixed height
            main_container.grid_columnconfigure(0, weight=1)

            self.logger.debug("Main container created")
        except Exception as e:
            self.logger.error(f"Error creating main container: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

        # Create header frame with proper background
        header_frame = tk.Frame(
            main_container, bg=self.gui_manager.theme_colors["background"], height=100
        )
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_propagate(False)

        # Language toggle in top right
        self.language_btn = self.gui_manager.create_modern_button(
            header_frame,
            text="Language",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._toggle_language,
        )
        self.language_btn.place(relx=0.98, rely=0.5, anchor="e")

        # Create centered container for logo and title
        center_container = tk.Frame(
            header_frame, bg=self.gui_manager.theme_colors["background"]
        )
        center_container.place(relx=0.5, rely=0.5, anchor="center")

        # Add logo and title side by side
        try:
            import PIL.Image
            import PIL.ImageTk

            # Get logo path
            logo_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "resources", "full_logo.png"
            )

            if os.path.exists(logo_path):
                # Load and resize logo to larger size
                logo_image = PIL.Image.open(logo_path)

                # Resize to larger height (e.g., 80 pixels for better visibility)
                logo_height = 80
                aspect_ratio = logo_image.width / logo_image.height
                logo_width = int(logo_height * aspect_ratio)
                logo_image = logo_image.resize(
                    (logo_width, logo_height), PIL.Image.Resampling.LANCZOS
                )

                # Convert to PhotoImage
                self.logo_photo = PIL.ImageTk.PhotoImage(logo_image)

                # Create label for logo
                logo_label = tk.Label(
                    center_container,
                    image=self.logo_photo,
                    bg=self.gui_manager.theme_colors["background"],
                )
                logo_label.pack(side=tk.LEFT, padx=(0, 20))

                # Add GeoVue title next to logo
                title_label = tk.Label(
                    center_container,
                    text="GeoVue",
                    font=("Arial", 24, "bold"),
                    bg=self.gui_manager.theme_colors["background"],
                    fg=self.gui_manager.theme_colors["text"],
                )
                title_label.pack(side=tk.LEFT)

                self.logger.debug(f"Logo and title added to first run dialog")

        except Exception as e:
            self.logger.debug(f"Could not load logo for first run dialog: {e}")
            # If logo fails, just show the title
            title_label = tk.Label(
                center_container,
                text="GeoVue",
                font=("Arial", 24, "bold"),
                bg=self.gui_manager.theme_colors["background"],
                fg=self.gui_manager.theme_colors["text"],
            )
            title_label.pack()

        # Create modern notebook for wizard-style interface
        self.notebook = ModernNotebook(
            main_container,
            theme_colors=self.gui_manager.theme_colors,
            fonts=self.gui_manager.fonts,
        )
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Create pages (include Local Storage but it won't be in the Next flow)
        self._create_welcome_page()
        self._create_setup_instructions_page()
        self._create_local_storage_page()  # Keep this for manual access
        self._create_backup_storage_page()
        self._create_summary_page()

        # Disable navigation to later pages
        self.notebook.tab(1, state="disabled")
        self.notebook.tab(2, state="disabled")
        self.notebook.tab(3, state="disabled")
        self.notebook.tab(4, state="disabled")

        # Button frame at bottom
        button_frame = ttk.Frame(main_container, padding="5")
        button_frame.grid(row=2, column=0, sticky="ew")

        # Create a separator above buttons
        separator = ttk.Separator(button_frame, orient="horizontal")
        separator.pack(fill=tk.X, pady=(0, 5))

        # Button container
        btn_container = ttk.Frame(button_frame)
        btn_container.pack(fill=tk.X)

        self.cancel_btn = self.gui_manager.create_modern_button(
            btn_container,
            text=DialogHelper.t("Cancel"),
            color=self.gui_manager.theme_colors["accent_red"],
            command=self._cancel,
        )
        self.cancel_btn.pack(side=tk.LEFT)

        self.back_btn = self.gui_manager.create_modern_button(
            btn_container,
            text=DialogHelper.t("Back"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._go_back,
        )
        self.back_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.back_btn.set_state("disabled")

        # Right side buttons
        self.finish_btn = self.gui_manager.create_modern_button(
            btn_container,
            text=DialogHelper.t("Finish"),
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._finish,
        )
        self.finish_btn.pack(side=tk.RIGHT)
        self.finish_btn.pack_forget()  # Hidden initially

        self.next_btn = self.gui_manager.create_modern_button(
            btn_container,
            text=DialogHelper.t("Next"),
            color=self.gui_manager.theme_colors[
                "accent_green"
            ],  # Green for Next button
            command=self._go_next,
        )
        self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Handle window close button
        self.dialog.protocol("WM_DELETE_WINDOW", self._cancel)

        # Update dialog size to fit content
        self.dialog.update_idletasks()

        # Set a minimum size for the dialog to prevent content being cut off
        self.dialog.minsize(850, 700)

        # Automatically check default folder if it exists
        default_path = Path(self.local_path_var.get())
        if default_path.exists():
            self.logger.info(f"Default local path exists: {default_path}")
            # Skip checking since we're not showing the local storage page
            self.local_existing = True  # Assume existing if path exists

        # Ensure dialog is visible
        self.logger.debug(f"Parent exists: {self.parent.winfo_exists()}")
        self.logger.debug(f"Parent visible: {self.parent.winfo_viewable()}")
        self.logger.debug(f"Parent geometry: {self.parent.winfo_geometry()}")

        self.logger.debug(f"Grab current: {self.dialog.grab_current()}")
        self.logger.debug("About to call wait_window()")

        # Center dialog with size constraints after all content is added
        DialogHelper.center_dialog(
            self.dialog, parent=None, max_width=1200, max_height=800
        )

        # Wait for dialog
        self.dialog.wait_window()

        # After dialog closes
        if self.parent:
            self.logger.debug("Deiconifying parent window after dialog closed")
            self.parent.deiconify()

        # Clean up reference
        if hasattr(self.parent, "_first_run_dialog"):
            delattr(self.parent, "_first_run_dialog")

        return self.result

    def _toggle_language(self):
        """Toggle between available languages."""
        # Get current language
        current_lang = (
            DialogHelper.translator.current_language
            if DialogHelper.translator
            else "en"
        )

        # Simple toggle between English and French (extend as needed)
        if current_lang == "en":
            new_lang = "fr"
        else:
            new_lang = "en"

        # Update language
        if DialogHelper.translator:
            DialogHelper.translator.set_language(new_lang)

            # Save language preference
            if (
                hasattr(self.gui_manager, "config_manager")
                and self.gui_manager.config_manager
            ):
                self.gui_manager.config_manager.set("language", new_lang)

            # Update all text in dialog
            self._update_all_text()

    def _update_all_text(self):
        """Update all text elements after language change."""
        # Update dialog title
        self.dialog.title(DialogHelper.t("Welcome to GeoVue"))

        # Update button texts
        self.cancel_btn.set_text(DialogHelper.t("Cancel"))
        self.back_btn.set_text(DialogHelper.t("Back"))
        self.finish_btn.set_text(DialogHelper.t("Finish"))
        self.next_btn.set_text(DialogHelper.t("Next"))

        # Update tab titles (5 tabs with new structure)
        self.notebook.tab_buttons[0]._label.config(text=DialogHelper.t("Welcome"))
        self.notebook.tab_buttons[1]._label.config(text=DialogHelper.t("Setup Instructions"))
        self.notebook.tab_buttons[2]._label.config(text=DialogHelper.t("Local Storage"))
        self.notebook.tab_buttons[3]._label.config(text=DialogHelper.t("OneDrive Folder"))
        self.notebook.tab_buttons[4]._label.config(text=DialogHelper.t("Summary"))

        # Update page content - store references to update labels
        if hasattr(self, "_translatable_widgets"):
            for widget_info in self._translatable_widgets:
                widget = widget_info["widget"]
                text_key = widget_info["text_key"]
                if widget.winfo_exists():
                    widget.config(text=DialogHelper.t(text_key))

        # Update field labels with translations
        if hasattr(self, "_translatable_fields"):
            for field_frame in self._translatable_fields:
                if hasattr(field_frame, "update_translation"):
                    field_frame.update_translation()

        # Update custom checkboxes with translations
        if hasattr(self, "_translatable_checkboxes"):
            for checkbox_frame in self._translatable_checkboxes:
                if hasattr(checkbox_frame, "update_translation"):
                    checkbox_frame.update_translation()

    def _create_welcome_page(self):
        """Create the welcome page."""
        page = ttk.Frame(
            self.notebook.page_container, padding="20", style="Content.TFrame"
        )
        self.notebook.add(page, text=DialogHelper.t("Welcome"))

        # Track translatable widgets
        self._translatable_widgets = []
        self._translatable_fields = []
        self._translatable_checkboxes = []

        # Center content vertically
        spacer_top = ttk.Frame(page)
        spacer_top.pack(expand=True, fill=tk.BOTH)

        # Content frame
        content = ttk.Frame(page)
        content.pack()

        # Welcome title
        title = ttk.Label(
            content,
            text=DialogHelper.t("Welcome to GeoVue"),
            font=self.gui_manager.fonts["title"],
            style="Content.TLabel",
        )
        title.pack(pady=(0, 10))
        self._translatable_widgets.append(
            {"widget": title, "text_key": "Welcome to GeoVue"}
        )

        # subtitle = ttk.Label(
        #     content,
        #     text=DialogHelper.t("Chip Tray Photo Processor"),
        #     font=self.gui_manager.fonts["subtitle"],
        #     foreground=self.gui_manager.theme_colors["text"],
        #     style='Content.TLabel'
        # )
        # subtitle.pack(pady=(0, 20))
        # self._translatable_widgets.append({'widget': subtitle, 'text_key': "Chip Tray Photo Processor"})

        # Welcome message
        welcome_text = (
            DialogHelper.t(
                "This setup wizard will help you configure the application for first use."
            )
            + "\n\n"
            + DialogHelper.t("You'll need to:")
        )

        welcome_label = ttk.Label(
            content,
            text=welcome_text,
            font=self.gui_manager.fonts["normal"],
            justify=tk.CENTER,
            style="Content.TLabel",
        )
        welcome_label.pack(pady=(0, 15))

        # Steps frame
        steps_frame = ttk.Frame(content)
        steps_frame.pack()

        steps = [
            "1. "
            + DialogHelper.t(
                "Synchronize the 'Chip Tray Photos' folder from Microsoft Teams to OneDrive"
            ),
            "2. "
            + DialogHelper.t(
                "Select your synchronized OneDrive folder location"
            ),
            "3. "
            + DialogHelper.t(
                "Set key folders to 'Always Keep on This Device' for offline access"
            ),
        ]

        for step in steps:
            step_label = ttk.Label(
                steps_frame,
                text=step,
                font=self.gui_manager.fonts["normal"],
                style="Content.TLabel",
            )
            step_label.pack(anchor=tk.W, pady=3)

        # Bottom text
        bottom_text = ttk.Label(
            content,
            text=DialogHelper.t("Click 'Next' to begin."),
            font=self.gui_manager.fonts["small"],
            foreground=self.gui_manager.theme_colors["text"],
            style="Content.TLabel",
        )
        bottom_text.pack(pady=(20, 0))
        self._translatable_widgets.append(
            {"widget": bottom_text, "text_key": "Click 'Next' to begin."}
        )

        spacer_bottom = ttk.Frame(page)
        spacer_bottom.pack(expand=True, fill=tk.BOTH)

    def _create_local_storage_page(self):
        """Create local storage selection page."""
        page = ttk.Frame(
            self.notebook.page_container, padding="20", style="Content.TFrame"
        )
        self.notebook.add(page, text=DialogHelper.t("Local Storage"))

        # Title
        title = ttk.Label(
            page,
            text=DialogHelper.t("Select Local Storage Location"),
            font=self.gui_manager.fonts["title"],
            style="Content.TLabel",
        )
        title.pack(pady=(0, 10))
        self._translatable_widgets.append(
            {"widget": title, "text_key": "Select Local Storage Location"}
        )

        # Instructions
        instructions = ttk.Label(
            page,
            text=DialogHelper.t(
                "Select where GeoVue should store and process images on this computer."
            )
            + "\n"
            + DialogHelper.t(
                "The required folder structure will be created automatically."
            ),
            font=self.gui_manager.fonts["normal"],
            justify=tk.CENTER,
            style="Content.TLabel",
        )
        instructions.pack(pady=(0, 20))

        # Path selection frame
        path_frame = ttk.LabelFrame(
            page, text=DialogHelper.t("Storage Location"), padding="15"
        )
        path_frame.pack(fill=tk.X, pady=10)

        self.local_path_var = tk.StringVar(value="C:\\GeoVue Chip Tray Photos")

        # Path input row
        path_row = ttk.Frame(path_frame)
        path_row.pack(fill=tk.X)

        # Use themed entry with GUIManager
        self.local_path_entry = tk.Entry(
            path_row,
            textvariable=self.local_path_var,
            font=self.gui_manager.fonts["normal"],
            state="readonly",
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            readonlybackground=self.gui_manager.theme_colors["field_bg"],
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
        )
        self.local_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = self.gui_manager.create_modern_button(
            path_row,
            text=DialogHelper.t("Browse"),
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._browse_local_folder,
        )
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Status frame for local folder check
        self.local_status_frame = ttk.LabelFrame(
            page, text=DialogHelper.t("Local Folder Status"), padding="10"
        )
        self.local_status_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        # Add scrollbar to status text
        status_container = ttk.Frame(self.local_status_frame)
        status_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(status_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Use themed text display
        self.local_status_text = self.gui_manager.create_text_display(
            status_container, height=10, wrap=tk.WORD, readonly=True
        )
        self.local_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.local_status_text.yview)
        self.local_status_text.config(yscrollcommand=scrollbar.set)

        # Initialize with default message
        self.local_status_text.config(state="normal")
        self.local_status_text.insert(
            tk.END, DialogHelper.t("Select a folder to check its structure.")
        )
        self.local_status_text.config(state="disabled")

    def _create_setup_instructions_page(self):
        """Create setup instructions page."""
        page = ttk.Frame(
            self.notebook.page_container, padding="20", style="Content.TFrame"
        )
        self.notebook.add(page, text=DialogHelper.t("Setup Instructions"))

        # Title
        title = ttk.Label(
            page,
            text=DialogHelper.t("Synchronize SharePoint Folder"),
            font=self.gui_manager.fonts["title"],
            style="Content.TLabel",
        )
        title.pack(pady=(0, 20))

        # Instructions Frame
        instructions_frame = ttk.LabelFrame(
            page, text=DialogHelper.t("Steps to Sync"), padding="15"
        )
        instructions_frame.pack(fill=tk.X, pady=(0, 15))

        steps = [
            "1. " + DialogHelper.t("You need to 'Sync' a SharePoint folder which contains the Chip Tray Photos folder."),
            "2. " + DialogHelper.t("Open Microsoft Teams and navigate to the channel containing the 'Chip Tray Photos' folder."),
            "3. " + DialogHelper.t("In the toolbar at the top, click 'Sync'. This will create a link in your File Explorer to the SharePoint location."),
        ]

        for step in steps:
            step_label = ttk.Label(
                instructions_frame,
                text=step,
                font=self.gui_manager.fonts["normal"],
                style="Content.TLabel",
                wraplength=750,
            )
            step_label.pack(anchor=tk.W, pady=5)

        # Required folders for offline access
        offline_frame = ttk.LabelFrame(
            page, text=DialogHelper.t("⚠️ Required: Set Folders for Offline Access"), padding="15"
        )
        offline_frame.pack(fill=tk.X, pady=(0, 15))
        
        offline_label = ttk.Label(
            offline_frame,
            text=DialogHelper.t("After sync completes, right-click these folders in OneDrive and select 'Always Keep on This Device':"),
            font=self.gui_manager.fonts["normal"],
            style="Content.TLabel",
            wraplength=750,
        )
        offline_label.pack(anchor=tk.W, pady=(0, 10))
        
        folders = [
            "• " + DialogHelper.t("Extracted Compartment Images (Required for viewing)"),
            "• " + DialogHelper.t("Chip Tray Register (Required for data)"),
            "• " + DialogHelper.t("Drillhole Datasets (Required for geological data)"),
            "• " + DialogHelper.t("Processed Original Images (Optional but recommended)"),
        ]
        
        for folder in folders:
            folder_label = ttk.Label(
                offline_frame,
                text=folder,
                font=self.gui_manager.fonts["normal"],
                style="Content.TLabel",
            )
            folder_label.pack(anchor=tk.W, pady=2, padx=(20, 0))
            
        warning = ttk.Label(
            offline_frame,
            text=DialogHelper.t("Note: Initial download may exceed 100GB. Ensure sufficient disk space."),
            font=self.gui_manager.fonts["normal"],
            foreground="orange",
            style="Content.TLabel",
        )
        warning.pack(anchor=tk.W, pady=(10, 0))

        # Bottom instruction
        bottom_text = ttk.Label(
            page,
            text=DialogHelper.t("Once you have completed these steps, click 'Next'."),
            font=self.gui_manager.fonts["normal"],
            style="Content.TLabel",
        )
        bottom_text.pack(pady=(20, 0))

    def _create_backup_storage_page(self):
        """Create backup/shared storage selection page."""
        page = ttk.Frame(
            self.notebook.page_container, padding="20", style="Content.TFrame"
        )
        self.notebook.add(page, text=DialogHelper.t("OneDrive Folder"))

        # Title
        title = ttk.Label(
            page,
            text=DialogHelper.t("Select OneDrive Folder"),
            font=self.gui_manager.fonts["title"],
            style="Content.TLabel",
        )
        title.pack(pady=(0, 20))

        # Path selection frame
        self.backup_frame = ttk.LabelFrame(
            page, text=DialogHelper.t("OneDrive Folder Location"), padding="15"
        )
        self.backup_frame.pack(fill=tk.X, pady=10)

        # Path input row
        path_row = ttk.Frame(self.backup_frame)
        path_row.pack(fill=tk.X)

        self.backup_path_entry = tk.Entry(
            path_row,
            textvariable=self.backup_path_var,
            font=self.gui_manager.fonts["normal"],
            state="readonly",
            bg=self.gui_manager.theme_colors["field_bg"],
            fg=self.gui_manager.theme_colors["text"],
            readonlybackground=self.gui_manager.theme_colors["field_bg"],
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
        )
        self.backup_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.backup_browse_btn = self.gui_manager.create_modern_button(
            path_row,
            text="Browse",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._browse_backup_folder,
        )
        self.backup_browse_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Hint
        hint_label = ttk.Label(
            self.backup_frame,
            text=DialogHelper.t("Look for: C:\\Users\\[YourName]\\OneDrive - Fortescue\\...\\Chip Tray Photos"),
            font=self.gui_manager.fonts["small"],
            foreground=self.gui_manager.theme_colors["accent_blue"],
            style="Content.TLabel",
        )
        hint_label.pack(anchor=tk.W, pady=(10, 0))

        # Status frame
        self.backup_status_frame = ttk.LabelFrame(
            page, text=DialogHelper.t("Folder Verification"), padding="10"
        )
        self.backup_status_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Status text with scrollbar
        status_container = ttk.Frame(self.backup_status_frame)
        status_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(status_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.backup_status_text = self.gui_manager.create_text_display(
            status_container, height=10, wrap=tk.WORD, readonly=True
        )
        self.backup_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.backup_status_text.yview)
        self.backup_status_text.config(yscrollcommand=scrollbar.set)
        
        # Initial message
        self.backup_status_text.config(state="normal")
        self.backup_status_text.insert(
            tk.END, DialogHelper.t("Please browse to select your synchronized 'Chip Tray Photos' folder.")
        )
        self.backup_status_text.config(state="disabled")

    def _create_summary_page(self):
        """Create summary page."""
        page = ttk.Frame(
            self.notebook.page_container, padding="20", style="Content.TFrame"
        )
        self.notebook.add(page, text=DialogHelper.t("Summary"))

        self.summary_content = ttk.Frame(page)
        self.summary_content.pack(fill=tk.BOTH, expand=True)

    def _populate_summary_page(self):
        """Populate the summary page with selected paths."""
        # Clear existing content
        for widget in self.summary_content.winfo_children():
            widget.destroy()

        # Title
        title = ttk.Label(
            self.summary_content,
            text=DialogHelper.t("Configuration Summary"),
            font=self.gui_manager.fonts["title"],
            style="Content.TLabel",
        )
        title.pack(pady=(0, 20))

        # Summary frame
        summary_frame = ttk.LabelFrame(
            self.summary_content, text=DialogHelper.t("Selected Paths"), padding="15"
        )
        summary_frame.pack(fill=tk.X, pady=(0, 15))

        # Configure grid columns
        summary_frame.grid_columnconfigure(1, weight=1)

        # Local path
        local_label = ttk.Label(
            summary_frame,
            text=DialogHelper.t("Local Storage") + ":",
            font=self.gui_manager.fonts["heading"],
            style="Content.TLabel",
        )
        local_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=5)

        local_value = ttk.Label(
            summary_frame,
            text=self.local_path_var.get(),
            font=self.gui_manager.fonts["normal"],
            style="Content.TLabel",
        )
        local_value.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Backup path
        backup_label = ttk.Label(
            summary_frame,
            text=DialogHelper.t("Backup/Shared Storage") + ":",
            font=self.gui_manager.fonts["heading"],
            style="Content.TLabel",
        )
        backup_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=5)

        backup_text = (
            self.backup_path_var.get()
            if self.use_backup_var.get()
            else DialogHelper.t("Not configured")
        )
        backup_value = ttk.Label(
            summary_frame,
            text=backup_text,
            font=self.gui_manager.fonts["normal"],
            foreground=(
                self.gui_manager.theme_colors["text"]
                if self.use_backup_var.get()
                else "gray"
            ),
            style="Content.TLabel",
        )
        backup_value.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Actions that will be taken
        actions_frame = ttk.LabelFrame(
            self.summary_content, text=DialogHelper.t("Setup Actions"), padding="15"
        )
        actions_frame.pack(fill=tk.X)

        actions = []
        
        # DEBUG: Check what values are set
        print(f"DEBUG: local_existing = {getattr(self, 'local_existing', 'NOT SET')}")
        print(f"DEBUG: backup_existing = {getattr(self, 'backup_existing', 'NOT SET')}")

        # Local folder actions
        if hasattr(self, "local_existing") and self.local_existing:
            action_text = "[OK] " + DialogHelper.t("Use existing local folder structure")
            print(f"DEBUG: Adding LOCAL action: {action_text}")
            actions.append(action_text)
        else:
            action_text = "[OK] " + DialogHelper.t("Create local folder structure")
            print(f"DEBUG: Adding LOCAL action: {action_text}")
            actions.append(action_text)

        # OneDrive folder actions (backup_existing tells us if OneDrive structure exists)
        if hasattr(self, "backup_existing") and self.backup_existing:
            action_text = "[OK] " + DialogHelper.t("Use existing OneDrive folder structure")
            print(f"DEBUG: Adding ONEDRIVE action: {action_text}")
            actions.append(action_text)
        else:
            action_text = "[OK] " + DialogHelper.t("Create missing folders in OneDrive")
            print(f"DEBUG: Adding ONEDRIVE action: {action_text}")
            actions.append(action_text)

        actions.append("[OK] " + DialogHelper.t("Save configuration"))

        for action in actions:
            action_label = ttk.Label(
                actions_frame,
                text=action,
                font=self.gui_manager.fonts["normal"],
                style="Content.TLabel",
            )
            action_label.pack(anchor=tk.W, pady=3)


    def _browse_local_folder(self):
        """Browse for local storage folder."""
        folder = filedialog.askdirectory(
            parent=self.dialog,  # Set parent for proper z-order
            title=DialogHelper.t("Select Local Storage Location"),
            initialdir=os.path.dirname(self.local_path_var.get()),
        )

        if folder:
            # Append GeoVue folder name to the selected path
            full_path = os.path.join(folder, "GeoVue Chip Tray Photos")
            self.local_path_var.set(full_path)
            self._check_local_folder_structure(full_path)

    def _browse_backup_folder(self):
        """Browse for backup/shared storage folder."""
        folder = filedialog.askdirectory(
            parent=self.dialog,  # Set parent for proper z-order
            title=DialogHelper.t("Select OneDrive or Network Folder for Backup"),
        )

        if folder:
            self.backup_path_var.set(folder)
            self._check_backup_folder_structure(folder)
            
            # Enable Next button after successful folder selection
            if hasattr(self, 'next_btn'):
                self.next_btn.set_state("normal")

    def _toggle_backup_controls(self):
        """Enable/disable backup path controls based on checkbox."""
        if self.use_backup_var.get():
            self.backup_path_entry.config(state="readonly")
            self.backup_browse_btn.set_state("normal")
            self.backup_status_text.config(state="normal")
            self.backup_status_text.delete(1.0, tk.END)
            self.backup_status_text.insert(
                tk.END, DialogHelper.t("Select a folder to check its structure.")
            )
            self.backup_status_text.config(state="disabled")
        else:
            self.backup_path_entry.config(state="disabled")
            self.backup_browse_btn.set_state("disabled")
            self.backup_status_text.config(state="normal")
            self.backup_status_text.delete(1.0, tk.END)
            self.backup_status_text.config(state="disabled")
            self.backup_path_var.set("")

    def _check_local_folder_structure(self, folder_path):
        """Check if local folder has required GeoVue structure."""
        self.local_status_text.config(state="normal")
        self.local_status_text.delete(1.0, tk.END)

        path = Path(folder_path)

        # Track what we find
        found_folders = 0
        # Local folders don't include register folder
        local_folders = {
            k: v for k, v in self.REQUIRED_FOLDERS.items() if k != "register"
        }
        total_folders = len(local_folders)
        missing_items = []

        # Check the selected folder for the required structure
        self.local_status_text.insert(
            tk.END, DialogHelper.t("Checking folder") + f": {path.name}\n\n", "info"
        )

        # Check main folders (excluding register for local)
        for key, folder_name in local_folders.items():
            folder_path = path / folder_name
            if folder_path.exists():
                self.local_status_text.insert(
                    tk.END,
                    f"[OK] " + DialogHelper.t("Found") + f": {folder_name}\n",
                    "success",
                )
                found_folders += 1

                # Check subfolders
                if key in self.REQUIRED_SUBFOLDERS:
                    for subfolder in self.REQUIRED_SUBFOLDERS[key]:
                        subfolder_path = folder_path / subfolder
                        if subfolder_path.exists():
                            self.local_status_text.insert(
                                tk.END, f"  - {subfolder}\n", "success"
                            )
                        else:
                            self.local_status_text.insert(
                                tk.END,
                                f"  X "
                                + DialogHelper.t("Missing")
                                + f": {subfolder}\n",
                                "warning",
                            )
                            missing_items.append(f"{folder_name}/{subfolder}")
            else:
                self.local_status_text.insert(
                    tk.END,
                    f"[X] " + DialogHelper.t("Missing") + f": {folder_name}\n",
                    "warning",
                )
                missing_items.append(folder_name)

        # Determine if we have an existing structure
        self.local_existing = found_folders > 0

        # Summary
        if found_folders == total_folders and not missing_items:
            self.local_status_text.insert(
                tk.END,
                "\n[COMPLETE] " + DialogHelper.t("Complete folder structure found!") + "\n",
                "success",
            )
            self.local_status_text.insert(
                tk.END, DialogHelper.t("Will use existing structure.") + "\n", "success"
            )
        elif found_folders > 0:
            percentage = (found_folders / total_folders) * 100
            self.local_status_text.insert(
                tk.END,
                f"\n[WARNING] "
                + DialogHelper.t("Found")
                + f" {found_folders}/{total_folders} "
                + DialogHelper.t("folders")
                + f" ({percentage:.0f}%)\n",
                "warning",
            )
            if missing_items:
                self.local_status_text.insert(
                    tk.END,
                    DialogHelper.t("Missing items will be created.") + "\n",
                    "warning",
                )
        else:
            self.local_status_text.insert(
                tk.END,
                "\n[INFO] "
                + DialogHelper.t("Empty folder - new structure will be created.")
                + "\n",
                "info",
            )

        # Configure text tags for colors
        self.local_status_text.tag_config(
            "success", foreground=self.gui_manager.theme_colors["accent_green"]
        )
        self.local_status_text.tag_config("warning", foreground="#ff9800")  # Orange
        self.local_status_text.tag_config(
            "info", foreground=self.gui_manager.theme_colors["accent_blue"]
        )

        self.local_status_text.config(state="disabled")

    def _check_backup_folder_structure(self, folder_path):
        """Check if backup folder has required GeoVue structure."""
        self.backup_status_text.config(state="normal")
        self.backup_status_text.delete(1.0, tk.END)

        path = Path(folder_path)

        # Track what we find
        found_folders = 0
        total_folders = len(self.REQUIRED_FOLDERS)
        missing_items = []

        # Simply check the selected folder for the required structure
        self.backup_status_text.insert(
            tk.END, DialogHelper.t("Checking folder") + f": {path.name}\n\n", "info"
        )

        # Check main folders
        for key, folder_name in self.REQUIRED_FOLDERS.items():
            folder_path = path / folder_name
            if folder_path.exists():
                self.backup_status_text.insert(
                    tk.END,
                    f"[OK] " + DialogHelper.t("Found") + f": {folder_name}\n",
                    "success",
                )
                found_folders += 1

                # Check subfolders
                if key in self.REQUIRED_SUBFOLDERS:
                    for subfolder in self.REQUIRED_SUBFOLDERS[key]:
                        subfolder_path = folder_path / subfolder
                        if subfolder_path.exists():
                            self.backup_status_text.insert(
                                tk.END, f"  - {subfolder}\n", "success"
                            )
                        else:
                            self.backup_status_text.insert(
                                tk.END,
                                f"  X "
                                + DialogHelper.t("Missing")
                                + f": {subfolder}\n",
                                "warning",
                            )
                            missing_items.append(f"{folder_name}/{subfolder}")
            else:
                self.backup_status_text.insert(
                    tk.END,
                    f"[X] " + DialogHelper.t("Missing") + f": {folder_name}\n",
                    "warning",
                )
                missing_items.append(folder_name)

        # Check for Excel register
        register_path = (
            path / self.REQUIRED_FOLDERS["register"] / "Chip_Tray_Register.xlsx"
        )
        if register_path.exists():
            self.backup_status_text.insert(
                tk.END,
                "\n[OK] " + DialogHelper.t("Found Excel register") + "\n",
                "success",
            )
        else:
            self.backup_status_text.insert(
                tk.END,
                "\n[X] " + DialogHelper.t("Excel register not found") + "\n",
                "warning",
            )

        # Determine if we have an existing structure
        self.backup_existing = found_folders > 0

        # Summary
        if found_folders == total_folders and not missing_items:
            self.backup_status_text.insert(
                tk.END,
                "\n[COMPLETE] " + DialogHelper.t("Complete folder structure found!") + "\n",
                "success",
            )
            self.backup_status_text.insert(
                tk.END, DialogHelper.t("Will use existing structure.") + "\n", "success"
            )
        elif found_folders > 0:
            percentage = (found_folders / total_folders) * 100
            self.backup_status_text.insert(
                tk.END,
                f"\n[WARNING] "
                + DialogHelper.t("Found")
                + f" {found_folders}/{total_folders} "
                + DialogHelper.t("folders")
                + f" ({percentage:.0f}%)\n",
                "warning",
            )
            if missing_items:
                self.backup_status_text.insert(
                    tk.END,
                    DialogHelper.t("Missing items will be created.") + "\n",
                    "warning",
                )
        else:
            self.backup_status_text.insert(
                tk.END,
                "\n[INFO] "
                + DialogHelper.t("Empty folder - new structure will be created.")
                + "\n",
                "info",
            )

        # Configure text tags for colors
        self.backup_status_text.tag_config(
            "success", foreground=self.gui_manager.theme_colors["accent_green"]
        )
        self.backup_status_text.tag_config("warning", foreground="#ff9800")  # Orange
        self.backup_status_text.tag_config(
            "info", foreground=self.gui_manager.theme_colors["accent_blue"]
        )

        self.backup_status_text.config(state="disabled")

    def _on_tab_changed(self, event):
        """Handle tab change events."""
        current_tab = self.notebook.index(self.notebook.select())

        # Update button visibility
        if current_tab == 0:  # Welcome
            self.back_btn.set_state("disabled")
            self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
            self.next_btn.set_state("normal")
            self.finish_btn.pack_forget()
        elif current_tab == 1:  # Setup Instructions
            self.back_btn.set_state("normal")
            self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
            self.next_btn.set_state("normal")
            self.finish_btn.pack_forget()
        elif current_tab == 2:  # Local Storage (optional)
            self.back_btn.set_state("normal")
            self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
            self.next_btn.set_state("normal")
            self.finish_btn.pack_forget()
        elif current_tab == 3:  # OneDrive Folder
            self.back_btn.set_state("normal")
            self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
            self.finish_btn.pack_forget()
            # Disable Next button if no backup path selected
            if not self.backup_path_var.get():
                self.next_btn.set_state("disabled")
            else:
                self.next_btn.set_state("normal")
        elif current_tab == 4:  # Summary
            self.back_btn.set_state("normal")
            self.next_btn.pack_forget()
            self.finish_btn.pack(side=tk.RIGHT)
            # Populate summary when we reach it
            self._populate_summary_page()

    def _go_next(self):
        """Go to next page."""
        current = self.notebook.index(self.notebook.select())

        # Determine next tab (skip Local Storage tab)
        if current == 0:  # Welcome -> Setup Instructions
            next_tab = 1
        elif current == 1:  # Setup Instructions -> OneDrive Folder (skip Local Storage)
            next_tab = 3
        elif current == 3:  # OneDrive Folder -> Summary
            next_tab = 4
        else:
            return  # Already at the end

        # Validate before proceeding
        if current == 3:  # OneDrive Folder page
            if not self.backup_path_var.get():
                DialogHelper.show_message(
                    self.dialog,
                    DialogHelper.t("Error"),
                    DialogHelper.t("Please select your synchronized OneDrive folder."),
                    message_type="error",
                )
                return

        # Enable and switch to next tab
        self.notebook.tab(next_tab, state="normal")
        self.notebook.select(next_tab)

    def _go_back(self):
        """Go to previous page."""
        current = self.notebook.index(self.notebook.select())
        if current > 0:
            self.notebook.select(current - 1)

    def _cancel(self):
        """Cancel setup."""
        self.logger.debug("Cancel button clicked")
        self.result = None
        self.dialog.destroy()

    def _finish(self):
        """Finish setup and create folders."""
        self.logger.debug("Finish button clicked")
        try:
            # Get the selected paths
            local_path = Path(self.local_path_var.get())
            backup_path = Path(self.backup_path_var.get())  # Always required now

            # Use FileManager to create folder structures
            if hasattr(self.gui_manager, 'file_manager') and self.gui_manager.file_manager:
                # Create local folder structure (FileManager handles skipping register for local)
                if not hasattr(self, 'local_existing') or not self.local_existing:
                    self.gui_manager.file_manager.create_local_folder_structure(str(local_path))
                
                # Create OneDrive folder structure if needed
                if backup_path:
                    if not hasattr(self, 'backup_existing') or not self.backup_existing:
                        self.gui_manager.file_manager.create_folder_structure(str(backup_path))
            else:
                self.logger.error("FileManager not available for folder creation")
                raise Exception("FileManager not available")

            # Prepare result
            self.result = {
                "storage_type": "both",  # Always both now
                "local_folder_path": str(local_path),
                "shared_folder_path": str(backup_path),
                "folder_paths": self._get_folder_paths(local_path, backup_path),
            }

            self.dialog.destroy()

        except Exception as e:
            DialogHelper.show_message(
                self.dialog,
                DialogHelper.t("Error"),
                DialogHelper.t("Failed to create folders") + f": {str(e)}",
                message_type="error",
            )

    def _create_folder_structure(self, base_path: Path, is_shared: bool = False):
        """Use FileManager to create the required folder structure."""
        if hasattr(self.gui_manager, 'file_manager') and self.gui_manager.file_manager:
            # Use FileManager's method to create folders
            if is_shared:
                # For shared/OneDrive, create all folders including register
                self.gui_manager.file_manager.create_folder_structure(str(base_path))
            else:
                # For local, FileManager will handle skipping the register folder
                self.gui_manager.file_manager.create_local_folder_structure(str(base_path))
        else:
            # Fallback if FileManager not available (shouldn't happen)
            self.logger.error("FileManager not available for folder creation")
            raise Exception("FileManager not available")

    def _get_folder_paths(self, local_path: Path, backup_path: Path = None) -> dict:
        """Get all folder paths for configuration."""
        paths = {
            "local_folder_path": str(local_path),
            "FileManager_output_directory": str(
                local_path
            ),  # For backward compatibility
            # Add all FileManager directory structure paths
            "register": str(local_path / self.REQUIRED_FOLDERS["register"]),
            "images_to_process": str(local_path / self.REQUIRED_FOLDERS["images"]),
            "processed_originals": str(local_path / self.REQUIRED_FOLDERS["processed"]),
            "approved_originals": str(
                local_path / self.REQUIRED_FOLDERS["processed"] / "Approved Originals"
            ),
            "rejected_originals": str(
                local_path / self.REQUIRED_FOLDERS["processed"] / "Rejected Originals"
            ),
            "chip_compartments": str(
                local_path / self.REQUIRED_FOLDERS["compartments"]
            ),
            "drill_traces": str(local_path / self.REQUIRED_FOLDERS["traces"]),
        }

        # Add backup/OneDrive paths if configured
        if backup_path:
            paths.update(
                {
                    "shared_folder_path": str(backup_path),
                    "shared_folder_approved_folder": str(
                        backup_path / self.REQUIRED_FOLDERS["compartments"]
                    ),
                    "shared_folder_processed_originals": str(
                        backup_path / self.REQUIRED_FOLDERS["processed"]
                    ),
                    "shared_folder_rejected_folder": str(
                        backup_path
                        / self.REQUIRED_FOLDERS["processed"]
                        / "Rejected Originals"
                    ),
                    "shared_folder_drill_traces": str(
                        backup_path / self.REQUIRED_FOLDERS["traces"]
                    ),
                    "shared_folder_register_path": str(
                        backup_path
                        / self.REQUIRED_FOLDERS["register"]
                        / "Chip_Tray_Register.xlsx"
                    ),
                    "shared_folder_register_data_folder": str(
                        backup_path
                        / self.REQUIRED_FOLDERS["register"]
                        / "Register Data (Do not edit)"
                    ),
                    # ADD: Missing compartment paths that OneDrivePathManager expects
                    "shared_folder_extracted_compartments_folder": str(
                        backup_path / self.REQUIRED_FOLDERS["compartments"]
                    ),
                    "shared_folder_approved_compartments_folder": str(
                        backup_path
                        / self.REQUIRED_FOLDERS["compartments"]
                        / "Approved Compartment Images"
                    ),
                    "shared_folder_review_compartments_folder": str(
                        backup_path
                        / self.REQUIRED_FOLDERS["compartments"]
                        / "Compartment Images for Review"
                    ),
                    "shared_folder_cross_sections": str(
                        backup_path / self.REQUIRED_FOLDERS["cross_sections"]
                    ),
                }
            )

        return paths