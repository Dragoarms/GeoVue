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
        if hasattr(gui_manager, 'file_manager') and gui_manager.file_manager:
            # Use FileManager's folder names if available
            self.REQUIRED_FOLDERS = {
                'register': 'Chip Tray Register',
                'images': 'Images to Process',
                'processed': 'Processed Original Images',
                'compartments': 'Extracted Compartment Images',
                'traces': 'Drillhole Traces',
                'temp_review': 'Compartments for Review',
                'debugging': 'Debugging'
            }
            self.REQUIRED_SUBFOLDERS = {
                'processed': ['Approved Originals', 'Rejected Originals'],
                'debugging': ['Blur Analysis', 'Debug Images'],
                'register': ['Register Data (Do not edit)']
            }
        else:
            # Fallback definitions
            self.REQUIRED_FOLDERS = {
                'register': 'Chip Tray Register',
                'images': 'Images to Process',
                'processed': 'Processed Original Images',
                'compartments': 'Extracted Compartment Images',
                'traces': 'Drillhole Traces',
                'temp_review': 'Compartments for Review',
                'debugging': 'Debugging'
            }
            self.REQUIRED_SUBFOLDERS = {
                'processed': ['Approved Originals', 'Rejected Originals'],
                'debugging': ['Blur Analysis', 'Debug Images'],
                'register': ['Register Data (Do not edit)']
            }
        
    def show(self) -> dict:
        """Show the first run dialog and return configuration."""
        # Create dialog without fixed size - let it size to content
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            "Welcome to GeoVue",
            modal=True,
            size_ratio=None,  # No fixed size ratio
            min_width=800,
            min_height=600
        )
        
        # Configure dialog background
        self.dialog.configure(bg=self.gui_manager.theme_colors["background"])
        
        # Set icon if available
        try:
            icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "logo.ico")
            if os.path.exists(icon_path):
                self.dialog.iconbitmap(icon_path)
        except Exception as e:
            self.logger.debug(f"Could not set icon: {e}")
        
        # Apply theme
        if self.gui_manager:
            self.gui_manager.apply_theme(self.dialog)
            # Configure ttk styles (includes notebook styles)
            self.gui_manager.configure_ttk_styles(self.dialog)
        
        # Create main_container first
        main_container = ttk.Frame(self.dialog, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create header frame with proper background
        header_frame = ttk.Frame(main_container, style='Content.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create centered container for logo and title
        center_container = ttk.Frame(header_frame, style='Content.TFrame')
        center_container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Add logo and title side by side
        try:
            import PIL.Image
            import PIL.ImageTk
            
            # Get logo path
            logo_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "resources", 
                "full_logo.png"
            )
            
            if os.path.exists(logo_path):
                # Load and resize logo to larger size
                logo_image = PIL.Image.open(logo_path)
                
                # Resize to larger height (e.g., 80 pixels for better visibility)
                logo_height = 80
                aspect_ratio = logo_image.width / logo_image.height
                logo_width = int(logo_height * aspect_ratio)
                logo_image = logo_image.resize(
                    (logo_width, logo_height), 
                    PIL.Image.Resampling.LANCZOS
                )
                
                # Convert to PhotoImage
                self.logo_photo = PIL.ImageTk.PhotoImage(logo_image)
                
                # Create label for logo
                logo_label = tk.Label(
                    center_container,
                    image=self.logo_photo,
                    bg=self.gui_manager.theme_colors["background"]
                )
                logo_label.pack(side=tk.LEFT, padx=(0, 20))
                
                # Add GeoVue title next to logo
                title_label = tk.Label(
                    center_container,
                    text="GeoVue",
                    font=("Arial", 24, "bold"),
                    bg=self.gui_manager.theme_colors["background"],
                    fg=self.gui_manager.theme_colors["text"]
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
                fg=self.gui_manager.theme_colors["text"]
            )
            title_label.pack()
        # Set minimum height for header to prevent overlap
        header_frame.configure(height=100)
        header_frame.pack_propagate(False)

        # Create modern notebook for wizard-style interface
        self.notebook = ModernNotebook(
            main_container,
            theme_colors=self.gui_manager.theme_colors,
            fonts=self.gui_manager.fonts
        )
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        
        # Create pages
        self._create_welcome_page()
        self._create_local_storage_page()
        self._create_backup_storage_page()
        self._create_summary_page()
        
        # THEN disable navigation to later pages
        self.notebook.tab(1, state='disabled')
        self.notebook.tab(2, state='disabled')
        self.notebook.tab(3, state='disabled')
        
        # Button frame at bottom with padding
        button_frame = ttk.Frame(self.dialog, padding="10")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)        
        # Create a separator above buttons
        separator = ttk.Separator(button_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=(0, 10))
        
        # Button container
        btn_container = ttk.Frame(button_frame)
        btn_container.pack(fill=tk.X)
        
        self.cancel_btn = self.gui_manager.create_modern_button(
            btn_container,
            text="Cancel",
            color=self.gui_manager.theme_colors["accent_red"],
            command=self._cancel
        )
        self.cancel_btn.pack(side=tk.LEFT)
        
        self.back_btn = self.gui_manager.create_modern_button(
            btn_container,
            text="Back",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._go_back
        )
        self.back_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.back_btn.set_state("disabled")
        
        # Right side buttons
        self.finish_btn = self.gui_manager.create_modern_button(
            btn_container,
            text="Finish",
            color=self.gui_manager.theme_colors["accent_green"],
            command=self._finish
        )
        self.finish_btn.pack(side=tk.RIGHT)
        self.finish_btn.pack_forget()  # Hidden initially
        
        self.next_btn = self.gui_manager.create_modern_button(
            btn_container,
            text="Next",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._go_next
        )
        self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Update dialog size to fit content
        self.dialog.update_idletasks()
        
        # Center dialog
        DialogHelper.center_dialog(self.dialog)
        
        # Wait for dialog
        self.dialog.wait_window()
        
        return self.result

                    
    def _create_welcome_page(self):
        """Create the welcome page."""
        page = ttk.Frame(self.dialog, padding="30", style='Content.TFrame')
        self.notebook.add(page, text="Welcome")
        
        # Center content vertically
        spacer_top = ttk.Frame(page)
        spacer_top.pack(expand=True, fill=tk.BOTH)
        
        # Content frame
        content = ttk.Frame(page)
        content.pack()
        
        # Welcome title
        title = ttk.Label(
            content,
            text="Welcome to GeoVue",
            font=self.gui_manager.fonts["title"],
            style='Title.TLabel'
        )
        title.pack(pady=(0, 10))
        
        subtitle = ttk.Label(
            content,
            text="Chip Tray Photo Processor",
            font=self.gui_manager.fonts["subtitle"],
            foreground=self.gui_manager.theme_colors["text"],
            style='Content.TLabel'
        )
        subtitle.pack(pady=(0, 30))
        
        # Welcome message
        welcome_text = (
            "This setup wizard will help you configure the application for first use.\n\n"
            "You'll need to:"
        )
        
        welcome_label = ttk.Label(
            content,
            text=welcome_text,
            font=self.gui_manager.fonts["normal"],
            justify=tk.CENTER,
            style='Content.TLabel'
        )
        welcome_label.pack(pady=(0, 20))
        
        # Steps frame
        steps_frame = ttk.Frame(content)
        steps_frame.pack()
        
        steps = [
            "1. Select a local folder where GeoVue will process images",
            "2. Optionally select a backup/shared folder (OneDrive or network)",
            "3. Confirm the folder structure creation"
        ]
        
        for step in steps:
            step_label = ttk.Label(
                steps_frame,
                text=step,
                font=self.gui_manager.fonts["normal"],
                style='Content.TLabel'
            )
            step_label.pack(anchor=tk.W, pady=5)
        
        # Bottom text
        bottom_text = ttk.Label(
            content,
            text="Click 'Next' to begin.",
            font=self.gui_manager.fonts["small"],
            foreground=self.gui_manager.theme_colors["text"],
            style='Content.TLabel'
        )
        bottom_text.pack(pady=(30, 0))
        
        spacer_bottom = ttk.Frame(page)
        spacer_bottom.pack(expand=True, fill=tk.BOTH)
        
    def _create_local_storage_page(self):
        """Create local storage selection page."""
        page = ttk.Frame(self.dialog, padding="30", style='Content.TFrame')
        self.notebook.add(page, text="Local Storage")
        
        # Title
        title = ttk.Label(
            page,
            text="Select Local Storage Location",
            font=self.gui_manager.fonts["title"],
            style='SectionTitle.TLabel'
        )
        title.pack(pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(
            page,
            text="Select where GeoVue should store and process images on this computer.\n"
                 "The required folder structure will be created automatically.",
            font=self.gui_manager.fonts["normal"],
            justify=tk.CENTER,
            style='Content.TLabel'
        )
        instructions.pack(pady=(0, 30))
        
        # Path selection frame
        path_frame = ttk.LabelFrame(page, text="Storage Location", padding="20")
        path_frame.pack(fill=tk.X, pady=10)
        
        self.local_path_var = tk.StringVar(value="C:\\GeoVue Chip Tray Photos")
        
        # Use themed field creation
        entry_frame, self.local_path_entry = self.gui_manager.create_field_with_label(
            path_frame,
            "",  # No label since we're in a LabelFrame
            self.local_path_var,
            field_type="entry",
            readonly=True,
            width=50
        )
        entry_frame.pack(fill=tk.X)
        
        browse_btn = self.gui_manager.create_modern_button(
            entry_frame,
            text="Browse",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._browse_local_folder
        )
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Info about what will be created
        info_frame = ttk.LabelFrame(page, text="Folders to be Created", padding="20")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Create two columns for folder list
        folders_container = ttk.Frame(info_frame)
        folders_container.pack(fill=tk.BOTH, expand=True)
        
        left_column = ttk.Frame(folders_container)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_column = ttk.Frame(folders_container)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        # Left column folders
        left_folders = [
            "📁 Chip Tray Register",
            "  └─ Register Data (Do not edit)",
            "📁 Images to Process",
            "📁 Processed Original Images",
            "  ├─ Approved Originals",
            "  └─ Rejected Originals"
        ]
        
        # Right column folders
        right_folders = [
            "📁 Extracted Compartment Images",
            "📁 Drillhole Traces",
            "📁 Debugging",
            "  ├─ Blur Analysis",
            "  └─ Debug Images"
        ]
        
        for folder in left_folders:
            label = ttk.Label(
                left_column,
                text=folder,
                font=self.gui_manager.fonts["code"],
                style='Content.TLabel'
            )
            label.pack(anchor=tk.W, pady=2)
            
        for folder in right_folders:
            label = ttk.Label(
                right_column,
                text=folder,
                font=self.gui_manager.fonts["code"],
                style='Content.TLabel'
            )
            label.pack(anchor=tk.W, pady=2)
        
    def _create_backup_storage_page(self):
        """Create backup/shared storage selection page."""
        page = ttk.Frame(self.dialog, padding="30", style='Content.TFrame')
        self.notebook.add(page, text="Backup Storage")
        
        # Title
        title = ttk.Label(
            page,
            text="Select Backup/Shared Storage (Optional)",
            font=self.gui_manager.fonts["title"],
            style='SectionTitle.TLabel'
        )
        title.pack(pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(
            page,
            text="Optionally select a OneDrive or network folder for backup and sharing.\n"
                 "This allows team collaboration and automatic backup of processed files.",
            font=self.gui_manager.fonts["normal"],
            justify=tk.CENTER,
            style='Content.TLabel'
        )
        instructions.pack(pady=(0, 20))
        
        # Checkbox to enable/disable backup
        self.use_backup_var = tk.BooleanVar(value=False)
        check_frame = ttk.Frame(page)
        check_frame.pack(pady=(0, 20))
        
        # Use custom checkbox
        backup_check = self.gui_manager.create_custom_checkbox(
            check_frame,
            text="Enable backup/shared storage",
            variable=self.use_backup_var,
            command=self._toggle_backup_controls
        )
        backup_check.pack()
        
        # Path selection frame
        self.backup_frame = ttk.LabelFrame(page, text="Backup Location", padding="20")
        self.backup_frame.pack(fill=tk.X, pady=10)
        
        self.backup_path_var = tk.StringVar()
        
        # Use themed field creation
        entry_frame, self.backup_path_entry = self.gui_manager.create_field_with_label(
            self.backup_frame,
            "",  # No label since we're in a LabelFrame
            self.backup_path_var,
            field_type="entry",
            readonly=True,
            width=50
        )
        entry_frame.pack(fill=tk.X)
        
        self.backup_browse_btn = self.gui_manager.create_modern_button(
            entry_frame,
            text="Browse",
            color=self.gui_manager.theme_colors["accent_blue"],
            command=self._browse_backup_folder
        )
        self.backup_browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        self.backup_browse_btn.set_state("disabled")
        
        # Status frame
        self.backup_status_frame = ttk.LabelFrame(
            page,
            text="Backup Folder Status",
            padding="10"
        )
        self.backup_status_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add scrollbar to status text
        status_container = ttk.Frame(self.backup_status_frame)
        status_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(status_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Use themed text display
        self.backup_status_text = self.gui_manager.create_text_display(
            status_container,
            height=10,
            wrap=tk.WORD,
            readonly=True
        )
        self.backup_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.backup_status_text.yview)
        self.backup_status_text.config(yscrollcommand=scrollbar.set)
        
    def _create_summary_page(self):
        """Create summary page."""
        page = ttk.Frame(self.dialog, padding="30", style='Content.TFrame')
        self.notebook.add(page, text="Summary")
        
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
            text="Configuration Summary",
            font=self.gui_manager.fonts["title"],
            style='SectionTitle.TLabel'
        )
        title.pack(pady=(0, 20))
        
        # Summary frame
        summary_frame = ttk.LabelFrame(self.summary_content, text="Selected Paths", padding="20")
        summary_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Configure grid columns
        summary_frame.grid_columnconfigure(1, weight=1)
        
        # Local path
        local_label = ttk.Label(
            summary_frame,
            text="Local Storage:",
            font=self.gui_manager.fonts["heading"],
            style='Content.TLabel'
        )
        local_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=10)
        
        local_value = ttk.Label(
            summary_frame,
            text=self.local_path_var.get(),
            font=self.gui_manager.fonts["normal"],
            style='Content.TLabel'
        )
        local_value.grid(row=0, column=1, sticky=tk.W, pady=10)
        
        # Backup path
        backup_label = ttk.Label(
            summary_frame,
            text="Backup/Shared Storage:",
            font=self.gui_manager.fonts["heading"],
            style='Content.TLabel'
        )
        backup_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=10)
        
        backup_text = self.backup_path_var.get() if self.use_backup_var.get() else "Not configured"
        backup_value = ttk.Label(
            summary_frame,
            text=backup_text,
            font=self.gui_manager.fonts["normal"],
            foreground=self.gui_manager.theme_colors["text"] if self.use_backup_var.get() else 'gray',
            style='Content.TLabel'
        )
        backup_value.grid(row=1, column=1, sticky=tk.W, pady=10)
        
        # Actions that will be taken
        actions_frame = ttk.LabelFrame(self.summary_content, text="Setup Actions", padding="20")
        actions_frame.pack(fill=tk.X)
        
        actions = ["✓ Create local folder structure"]
        if self.use_backup_var.get() and hasattr(self, 'backup_existing'):
            if self.backup_existing:
                actions.append("✓ Use existing backup folder structure")
            else:
                actions.append("✓ Create backup folder structure")
        
        actions.append("✓ Save configuration")
        
        for action in actions:
            action_label = ttk.Label(
                actions_frame,
                text=action,
                font=self.gui_manager.fonts["normal"],
                style='Content.TLabel'
            )
            action_label.pack(anchor=tk.W, pady=5)
        
    def _browse_local_folder(self):
        """Browse for local storage folder."""
        folder = filedialog.askdirectory(
            parent=self.dialog,  # Set parent for proper z-order
            title="Select Local Storage Location",
            initialdir=os.path.dirname(self.local_path_var.get())
        )
        
        if folder:
            # Append GeoVue folder name to the selected path
            full_path = os.path.join(folder, "GeoVue Chip Tray Photos")
            self.local_path_var.set(full_path)
            
    def _browse_backup_folder(self):
        """Browse for backup/shared storage folder."""
        folder = filedialog.askdirectory(
            parent=self.dialog,  # Set parent for proper z-order
            title="Select OneDrive or Network Folder for Backup"
        )
        
        if folder:
            self.backup_path_var.set(folder)
            self._check_backup_folder_structure(folder)
            
    def _toggle_backup_controls(self):
        """Enable/disable backup path controls based on checkbox."""
        if self.use_backup_var.get():
            self.backup_path_entry.config(state='readonly')
            self.backup_browse_btn.set_state("normal")
            self.backup_status_text.config(state='normal')
            self.backup_status_text.delete(1.0, tk.END)
            self.backup_status_text.insert(tk.END, "Select a folder to check its structure.")
            self.backup_status_text.config(state='disabled')
        else:
            self.backup_path_entry.config(state='disabled')
            self.backup_browse_btn.set_state("disabled")
            self.backup_status_text.config(state='normal')
            self.backup_status_text.delete(1.0, tk.END)
            self.backup_status_text.config(state='disabled')
            self.backup_path_var.set("")
            
    def _check_backup_folder_structure(self, folder_path):
        """Check if backup folder has required GeoVue structure."""
        self.backup_status_text.config(state='normal')
        self.backup_status_text.delete(1.0, tk.END)
        
        path = Path(folder_path)
        
        # Track what we find
        found_folders = 0
        total_folders = len(self.REQUIRED_FOLDERS)
        missing_items = []
        
        # Simply check the selected folder for the required structure
        self.backup_status_text.insert(tk.END, f"Checking folder: {path.name}\n\n", "info")
        
        # Check main folders
        for key, folder_name in self.REQUIRED_FOLDERS.items():
            folder_path = path / folder_name
            if folder_path.exists():
                self.backup_status_text.insert(tk.END, f"✓ Found: {folder_name}\n", "success")
                found_folders += 1
                
                # Check subfolders
                if key in self.REQUIRED_SUBFOLDERS:
                    for subfolder in self.REQUIRED_SUBFOLDERS[key]:
                        subfolder_path = folder_path / subfolder
                        if subfolder_path.exists():
                            self.backup_status_text.insert(tk.END, f"  ✓ {subfolder}\n", "success")
                        else:
                            self.backup_status_text.insert(tk.END, f"  ✗ Missing: {subfolder}\n", "warning")
                            missing_items.append(f"{folder_name}/{subfolder}")
            else:
                self.backup_status_text.insert(tk.END, f"✗ Missing: {folder_name}\n", "warning")
                missing_items.append(folder_name)
        
        # Check for Excel register
        register_path = path / self.REQUIRED_FOLDERS['register'] / "Chip_Tray_Register.xlsx"
        if register_path.exists():
            self.backup_status_text.insert(tk.END, "\n✓ Found Excel register\n", "success")
        else:
            self.backup_status_text.insert(tk.END, "\n✗ Excel register not found\n", "warning")
        
        # Determine if we have an existing structure
        self.backup_existing = found_folders > 0
        
        # Summary
        if found_folders == total_folders and not missing_items:
            self.backup_status_text.insert(tk.END, "\n✅ Complete folder structure found!\n", "success")
            self.backup_status_text.insert(tk.END, "Will use existing structure.\n", "success")
        elif found_folders > 0:
            percentage = (found_folders / total_folders) * 100
            self.backup_status_text.insert(tk.END, f"\n⚠ Found {found_folders}/{total_folders} folders ({percentage:.0f}%)\n", "warning")
            if missing_items:
                self.backup_status_text.insert(tk.END, "Missing items will be created.\n", "warning")
        else:
            self.backup_status_text.insert(tk.END, "\n📁 Empty folder - new structure will be created.\n", "info")
        
        # Configure text tags for colors
        self.backup_status_text.tag_config("success", foreground=self.gui_manager.theme_colors["accent_green"])
        self.backup_status_text.tag_config("warning", foreground="#ff9800")  # Orange
        self.backup_status_text.tag_config("info", foreground=self.gui_manager.theme_colors["accent_blue"])
        
        self.backup_status_text.config(state='disabled')
        
    def _on_tab_changed(self, event):
        """Handle tab change events."""
        current_tab = self.notebook.index(self.notebook.select())
        
        # Update button visibility
        if current_tab == 0:  # Welcome
            self.back_btn.set_state("disabled")
            self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
            self.finish_btn.pack_forget()
        elif current_tab == 1:  # Local Storage
            self.back_btn.set_state("normal")
            self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
            self.finish_btn.pack_forget()
        elif current_tab == 2:  # Backup Storage
            self.back_btn.set_state("normal")
            self.next_btn.pack(side=tk.RIGHT, padx=(0, 10))
            self.finish_btn.pack_forget()
        elif current_tab == 3:  # Summary
            self.back_btn.set_state("normal")
            self.next_btn.pack_forget()
            self.finish_btn.pack(side=tk.RIGHT)
            # Populate summary when we reach it
            self._populate_summary_page()
            
    def _go_next(self):
        """Go to next page."""
        current = self.notebook.index(self.notebook.select())
        
        # Validate current page before proceeding
        if current == 1:  # Local Storage page
            if not self.local_path_var.get():
                DialogHelper.show_message(
                    self.dialog,
                    "Error",
                    "Please select a local storage location.",
                    message_type="error"
                )
                return
        
        if current < 3:
            # Enable the next tab before switching
            self.notebook.tab(current + 1, state='normal')
            # Now switch to it
            self.notebook.select(current + 1)
    
    def _go_back(self):
        """Go to previous page."""
        current = self.notebook.index(self.notebook.select())
        if current > 0:
            self.notebook.select(current - 1)
    
    def _cancel(self):
        """Cancel setup."""
        self.result = None
        self.dialog.destroy()
    
    def _finish(self):
        """Finish setup and create folders."""
        try:
            # Get the selected paths
            local_path = Path(self.local_path_var.get())
            backup_path = Path(self.backup_path_var.get()) if self.use_backup_var.get() else None
            
            # Create local folder structure
            self._create_folder_structure(local_path)
            
            # Handle backup path - use it directly, don't look for "GeoVue Chip Tray Photos"
            if backup_path:
                # Create structure if needed
                if not self.backup_existing:
                    self._create_folder_structure(backup_path)
            
            # Prepare result
            self.result = {
                'storage_type': 'both' if backup_path else 'local',
                'local_folder_path': str(local_path),
                'shared_folder_path': str(backup_path) if backup_path else None,
                'folder_paths': self._get_folder_paths(local_path, backup_path)
            }
            
            self.dialog.destroy()
            
        except Exception as e:
            DialogHelper.show_message(
                self.dialog,
                "Error",
                f"Failed to create folders: {str(e)}",
                message_type="error"
            )
    
    def _create_folder_structure(self, base_path: Path):
        """Create the required folder structure."""
        # Create main folders
        for key, folder_name in self.REQUIRED_FOLDERS.items():
            folder_path = base_path / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Create subfolders
            if key in self.REQUIRED_SUBFOLDERS:
                for subfolder in self.REQUIRED_SUBFOLDERS[key]:
                    (folder_path / subfolder).mkdir(exist_ok=True)
        
        # Create register if it doesn't exist
        register_dir = base_path / self.REQUIRED_FOLDERS['register']
        if not (register_dir / "Chip_Tray_Register.xlsx").exists():

            # Create the JSON manager, it will handle template copy
            # Initialize JSON register manager to create Excel from template and data files
            json_manager = JSONRegisterManager(str(register_dir), self.logger)
    
    def _get_folder_paths(self, local_path: Path, backup_path: Path = None) -> dict:
            """Get all folder paths for configuration."""
            # ===================================================
            # MODIFIED: Include all FileManager paths
            # ===================================================
            paths = {
                'local_folder_path': str(local_path),
                'FileManager_output_directory': str(local_path),  # For backward compatibility
                
                # Add all FileManager directory structure paths
                'register': str(local_path / self.REQUIRED_FOLDERS['register']),
                'images_to_process': str(local_path / self.REQUIRED_FOLDERS['images']),
                'processed_originals': str(local_path / self.REQUIRED_FOLDERS['processed']),
                'approved_originals': str(local_path / self.REQUIRED_FOLDERS['processed'] / 'Approved Originals'),
                'rejected_originals': str(local_path / self.REQUIRED_FOLDERS['processed'] / 'Rejected Originals'),
                'chip_compartments': str(local_path / self.REQUIRED_FOLDERS['compartments']),
                'temp_review': str(local_path / self.REQUIRED_FOLDERS['temp_review']),
                'drill_traces': str(local_path / self.REQUIRED_FOLDERS['traces']),
                'debugging': str(local_path / self.REQUIRED_FOLDERS['debugging']),
                'blur_analysis': str(local_path / self.REQUIRED_FOLDERS['debugging'] / 'Blur Analysis'),
                'debug_images': str(local_path / self.REQUIRED_FOLDERS['debugging'] / 'Debug Images')
            }
            
            # Add backup/OneDrive paths if configured
            if backup_path:
                paths.update({
                    'shared_folder_path': str(backup_path),
                    'onedrive_approved_folder': str(backup_path / self.REQUIRED_FOLDERS['compartments']),
                    'onedrive_processed_originals': str(backup_path / self.REQUIRED_FOLDERS['processed']),
                    'onedrive_rejected_folder': str(backup_path / self.REQUIRED_FOLDERS['processed'] / 'Rejected Originals'),
                    'onedrive_drill_traces': str(backup_path / self.REQUIRED_FOLDERS['traces']),
                    'onedrive_register_path': str(backup_path / self.REQUIRED_FOLDERS['register'] / 'Chip_Tray_Register.xlsx'),
                    'onedrive_register_data_folder': str(backup_path / self.REQUIRED_FOLDERS['register'] / 'Register Data (Do not edit)')
                })
            
            return paths