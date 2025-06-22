"""
Progress Dialog for displaying background task progress.

This module provides a reusable progress dialog that can be used to show
progress for long-running operations while keeping the GUI responsive.

Author: George Symonds
Created: 2025
"""

import tkinter as tk
from tkinter import ttk
import threading
from typing import Optional, Callable
from gui.dialog_helper import DialogHelper


class ProgressDialog:
    """
    A modal progress dialog for showing progress of background operations.
    
    This dialog blocks user interaction with the main window while showing
    progress updates from a background thread.
    """
    
    def __init__(self, parent, title: str, message: str = ""):
        """
        Initialize the progress dialog.
        
        Args:
            parent: Parent window
            title: Dialog title
            message: Initial message to display
        """
        self.parent = parent
        self.title = title
        self.message = message
        self.dialog = None
        self.progress_var = tk.DoubleVar(value=0)
        self.message_var = tk.StringVar(value=message)
        self.cancelled = False
        self._create_dialog()
        
    def _create_dialog(self):
        """Create the progress dialog window."""
        # Create dialog using DialogHelper without fixed size
        self.dialog = DialogHelper.create_dialog(
            self.parent,
            self.title,
            modal=True
            # Remove size_ratio, min_width, min_height - let it auto-size
        )
        
        # Prevent closing
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Message label
        self.message_label = ttk.Label(
            main_frame,
            textvariable=self.message_var,
            wraplength=350
        )
        self.message_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100,
            length=350,
            mode='determinate'
        )
        self.progress_bar.pack(pady=(0, 10))
        
        # Percentage label
        self.percent_label = ttk.Label(
            main_frame,
            text="0%"
        )
        self.percent_label.pack()
        
        # Update and center the dialog after content is added
        self.dialog.update_idletasks()
        
        # Center on parent
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Get the actual size after packing
        self.dialog.update_idletasks()
        width = self.dialog.winfo_reqwidth()
        height = self.dialog.winfo_reqheight()
        
        # Center on screen
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2
        
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

    def _update_progress_ui(self, message: str, percentage: float):
        """Update UI elements on main thread."""
        try:
            self.message_var.set(message)
            self.progress_var.set(percentage)
            self.percent_label.config(text=f"{int(percentage)}%")
            self.dialog.update()
            
            # Auto-close if at 100%
            if percentage >= 100:
                self.dialog.after(500, self.close)  # Close after 500ms delay
        except Exception:
            pass  # Dialog may have been destroyed

    def update_progress(self, message: str, percentage: float):
        """
        Update progress from any thread.
        
        Args:
            message: Progress message
            percentage: Progress percentage (0-100)
        """
        if self.dialog and self.dialog.winfo_exists():
            # Schedule update on main thread
            self.dialog.after(0, self._update_progress_ui, message, percentage)
            
    def _update_progress_ui(self, message: str, percentage: float):
        """Update UI elements on main thread."""
        try:
            self.message_var.set(message)
            self.progress_var.set(percentage)
            self.percent_label.config(text=f"{int(percentage)}%")
            self.dialog.update()
        except Exception:
            pass  # Dialog may have been destroyed
            
    def close(self):
        """Close the progress dialog."""
        if self.dialog and self.dialog.winfo_exists():
            self.dialog.destroy()
            
    def run_with_progress(self, task: Callable, *args, **kwargs):
        """
        Run a task in background thread while showing progress.
        
        Args:
            task: Callable to run in background
            *args: Arguments to pass to task
            **kwargs: Keyword arguments to pass to task
            
        Returns:
            The result of the task
        """
        result = None
        exception = None
        
        def run_task():
            nonlocal result, exception
            try:
                result = task(*args, **kwargs)
            except Exception as e:
                exception = e
            finally:
                # Close dialog on main thread
                self.dialog.after(0, self.close)
                
        # Start task in background thread
        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()
        
        # Show dialog and wait
        self.dialog.wait_window()
        
        # Re-raise any exception that occurred
        if exception:
            raise exception
            
        return result