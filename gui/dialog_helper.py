# gui/dialog_helper.py

"""
    Utility class for creating consistent, properly positioned dialogs
    that remain on top and correctly capture focus. Includes translation support.
    Now with theme support for consistent styling across the application.
"""

import tkinter as tk
from tkinter import messagebox, ttk
import threading
import logging

# Create a logger for tracking thread issues
logger = logging.getLogger(__name__)

class DialogHelper:
    # Class-level references
    translator = None
    gui_manager = None
    
    @classmethod
    def set_translator(cls, translator):
        """Set the translation manager."""
        cls.translator = translator
    
    @classmethod
    def set_gui_manager(cls, gui_manager):
        """Set the GUI manager for theming."""
        cls.gui_manager = gui_manager
    
    @classmethod
    def t(cls, text):
        """
        Translate text using the class translator.

        Args:
            text (str): The text key to translate

        Returns:
            str: The translated text or original text if no translation available
        """
        if cls.translator.has_translation(text):
            return cls.translator.translate(text)
        else:
            if hasattr(cls, 'logger') and cls.logger:
                cls.logger.warning(f"Missing Translation: '{text}'")
            else:
                print(f"DEBUG: Missing Translation: '{text}'")
            return text


    
    @staticmethod
    def _check_main_thread(method_name):
        """
        Check if the current thread is the main thread and log a warning if not.
        
        Args:
            method_name: Name of the method being called for error messages
        
        Returns:
            True if on main thread, False otherwise
        """
        current_thread = threading.current_thread()
        is_main_thread = current_thread is threading.main_thread()
        
        # Always log which thread is calling each DialogHelper method for debugging
        logger.debug(f"DialogHelper.{method_name} called from thread: {current_thread.name} (main={is_main_thread})")
        
        if not is_main_thread:
            # Log a detailed warning with stack trace for debugging
            import traceback
            stack_trace = ''.join(traceback.format_stack())
            logger.error(f"⚠️ THREAD SAFETY ERROR: DialogHelper.{method_name} called from non-main thread!\n"
                        f"Current thread: {current_thread.name}\n"
                        f"Stack trace:\n{stack_trace}")
            return False
        return True


    @staticmethod
    def create_dialog(parent, title, modal=True, topmost=True, size_ratio=0.8, min_width=400, min_height=300, max_width=None, max_height=None):
        """
        Create a properly configured dialog window with theme styling.
        
        Args:
            parent: Parent window
            title: Dialog title
            modal: Whether dialog is modal
            topmost: Whether dialog stays on top
            size_ratio: Size ratio relative to screen
            min_width: Minimum width in pixels
            min_height: Minimum height in pixels
            
        Returns:
            Configured dialog window
        """
        print(f"DEBUG: DialogHelper.create_dialog called from thread: {threading.current_thread().name}")
        logger.debug(f"DialogHelper.create_dialog called with title: {title}")
        
        # Check thread safety
        DialogHelper._check_main_thread("create_dialog")
        
        dialog = tk.Toplevel(parent)
        dialog.title(DialogHelper.t(title))
        
        # Apply theme colors if gui_manager is available
        if DialogHelper.gui_manager and hasattr(DialogHelper.gui_manager, 'theme_colors'):
            theme_colors = DialogHelper.gui_manager.theme_colors
            dialog.configure(bg=theme_colors["background"])
        
        # Set window behavior
        if parent:
            dialog.transient(parent)  # Make dialog transient to parent
        
        if modal:
            dialog.grab_set()  # Make dialog modal
        
        dialog.focus_force()   # Force focus on dialog
        
        if topmost:
            dialog.attributes('-topmost', True)  # Keep on top
        
        # Ensure the dialog is in the normal state (not withdrawn)
        dialog.state('normal')
        
        # Ensure dialog gets focus and is visible
        dialog.lift()
        dialog.update()
        
        DialogHelper.center_dialog(dialog, size_ratio, min_width, min_height, max_width, max_height)        
        return dialog
    
    @staticmethod
    def create_error_dialog(parent, title, message):
        """
        Create a properly configured error dialog with themed components.
        
        Args:
            parent: Parent window
            title: Dialog title
            message: Error message
            
        Returns:
            Reference to the error dialog
        """
        # Check thread safety
        DialogHelper._check_main_thread("create_error_dialog")
        
        # Create base dialog with appropriate styling for errors
        dialog = DialogHelper.create_dialog(
            parent, 
            DialogHelper.t(title), 
            modal=True, 
            topmost=True,
            size_ratio=0.3,  # Smaller ratio for error dialogs
            min_width=350,
            min_height=180
        )
        
        # Apply theme colors if gui_manager is available
        theme_colors = None
        if DialogHelper.gui_manager and hasattr(DialogHelper.gui_manager, 'theme_colors'):
            theme_colors = DialogHelper.gui_manager.theme_colors
            dialog.configure(bg=theme_colors["background"])
        
        # Create message with error styling
        frame = tk.Frame(dialog, bg=theme_colors["background"] if theme_colors else None)
        frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        message_label = tk.Label(
            frame,
            text=DialogHelper.t(message),
            wraplength=300,
            justify=tk.CENTER,
            fg=theme_colors.get("accent_red", "red") if theme_colors else "red",
            bg=theme_colors["background"] if theme_colors else None,
            font=DialogHelper.gui_manager.fonts["normal"] if DialogHelper.gui_manager else None
        )
        message_label.pack(expand=True, pady=10)
        
        # Use ModernButton if gui_manager is available, otherwise use ttk.Button
        if DialogHelper.gui_manager and theme_colors:
            from gui.widgets.modern_button import ModernButton
            
            # OK button
            ok_button = ModernButton(
                dialog,
                text=DialogHelper.t("OK"),
                color=theme_colors["accent_blue"],
                command=dialog.destroy,
                theme_colors=theme_colors
            )
            ok_button.pack(pady=(0, 20))
        else:
            # Fallback to ttk button
            ok_button = ttk.Button(
                dialog,
                text=DialogHelper.t("OK"),
                command=dialog.destroy
            )
            ok_button.pack(pady=(0, 20))
        
        # Bind Escape and Enter keys to close the dialog
        dialog.bind("<Return>", lambda e: dialog.destroy())
        dialog.bind("<Escape>", lambda e: dialog.destroy())
        
        return dialog

    @staticmethod
    def create_message_dialog(parent, title, message, button_text="OK"):
        """
        Create a simple message dialog with themed components.
        
        Args:
            parent: Parent window
            title: Dialog title
            message: Message to display
            button_text: Text for the OK button
            
        Returns:
            Dialog window
        """
        # Check thread safety
        DialogHelper._check_main_thread("create_message_dialog")
        
        # Create base dialog
        dialog = DialogHelper.create_dialog(
            parent, DialogHelper.t(title), modal=True, topmost=True, 
            size_ratio=0.4, min_width=300, min_height=150
        )
        
        # Apply theme colors if gui_manager is available
        theme_colors = None
        if DialogHelper.gui_manager and hasattr(DialogHelper.gui_manager, 'theme_colors'):
            theme_colors = DialogHelper.gui_manager.theme_colors
            dialog.configure(bg=theme_colors["background"])
        
        # Create frame for content
        frame = tk.Frame(dialog, bg=theme_colors["background"] if theme_colors else None)
        frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Add message
        message_label = tk.Label(
            frame,
            text=DialogHelper.t(message),
            wraplength=dialog.winfo_width() - 60,  # Account for padding
            justify=tk.CENTER,
            bg=theme_colors["background"] if theme_colors else None,
            fg=theme_colors["text"] if theme_colors else None,
            font=DialogHelper.gui_manager.fonts["normal"] if DialogHelper.gui_manager else None
        )
        message_label.pack(expand=True, pady=10)
        
        # Use ModernButton if gui_manager is available, otherwise use ttk.Button
        if DialogHelper.gui_manager and theme_colors:
            from gui.widgets.modern_button import ModernButton
            
            # OK button
            ok_button = ModernButton(
                dialog,
                text=DialogHelper.t(button_text),
                color=theme_colors["accent_blue"],
                command=dialog.destroy,
                theme_colors=theme_colors
            )
            ok_button.pack(pady=(0, 20))
        else:
            # Fallback to ttk button
            ok_button = ttk.Button(
                dialog,
                text=DialogHelper.t(button_text),
                command=dialog.destroy
            )
            ok_button.pack(pady=(0, 20))
        
        return dialog
    
    @staticmethod
    def show_message(parent, title, message, message_type="info"):
        """Show a message dialog with an OK button."""
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.transient(parent)
        dialog.grab_set()
        
        # Configure dialog to be more compact
        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Set icon based on message type
        icon_text = "ℹ️"  # Default info icon
        if message_type == "error":
            icon_text = "❌"
        elif message_type == "warning":
            icon_text = "⚠️"
        elif message_type == "success":
            icon_text = "✅"
        
        # Icon and message in a horizontal layout
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 15))
        
        icon_label = ttk.Label(top_frame, text=icon_text, font=("Arial", 24))
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Message with proper wraplength
        screen_width = dialog.winfo_screenwidth()
        message_width = min(400, int(screen_width * 0.3))  # Cap at 400px or 30% of screen width
        
        msg_label = ttk.Label(
            top_frame, 
            text=message, 
            wraplength=message_width,
            justify=tk.LEFT
        )
        msg_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack()
        
        # Use modern button if gui_manager available
        gui_manager = DialogHelper.gui_manager
        if gui_manager:
            ok_button = gui_manager.create_modern_button(
                button_frame,
                text=DialogHelper.t("OK"),
                color=gui_manager.theme_colors["accent_blue"],
                command=dialog.destroy
            )
            ok_button.pack(padx=5, pady=5)
        else:
            ok_button = ttk.Button(button_frame, text=DialogHelper.t("OK"), command=dialog.destroy)
            ok_button.pack(padx=5, pady=5)
        
        ok_button.focus_set()
        
        # Center dialog on parent
        dialog.update_idletasks()  # Make sure dialog is updated for size calculations
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Handle dialog close via window manager
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        
        # Wait for dialog to close
        parent.wait_window(dialog)
    
    @staticmethod
    def confirm_dialog(parent, title, message, yes_text="Yes", no_text="No"):
        """Show a confirmation dialog with Yes/No buttons."""
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.transient(parent)
        dialog.grab_set()
        
        # Configure dialog to be more compact
        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Message with proper wraplength for readability
        screen_width = dialog.winfo_screenwidth()
        message_width = min(400, int(screen_width * 0.3))  # Cap at 400px or 30% of screen width
        
        msg_label = ttk.Label(
            main_frame, 
            text=message, 
            wraplength=message_width,
            justify=tk.CENTER
        )
        msg_label.pack(pady=(0, 15))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack()
        
        # Variable to store result
        result = [False]  # Use list for mutable reference
        
        # Callback functions
        def on_yes():
            result[0] = True
            dialog.destroy()
            
        def on_no():
            result[0] = False
            dialog.destroy()
        
        # Use modern buttons if gui_manager available
        gui_manager = DialogHelper.gui_manager
        if gui_manager:
            yes_button = gui_manager.create_modern_button(
                button_frame,
                text=yes_text,
                color=gui_manager.theme_colors["accent_green"],
                command=on_yes
            )
            yes_button.pack(side=tk.LEFT, padx=5)
            
            no_button = gui_manager.create_modern_button(
                button_frame,
                text=no_text,
                color=gui_manager.theme_colors["accent_red"],
                command=on_no
            )
            no_button.pack(side=tk.LEFT, padx=5)
        else:
            # Fallback to ttk buttons
            yes_button = ttk.Button(button_frame, text=yes_text, command=on_yes)
            yes_button.pack(side=tk.LEFT, padx=5)
            
            no_button = ttk.Button(button_frame, text=no_text, command=on_no)
            no_button.pack(side=tk.LEFT, padx=5)
        
        # Set focus to "No" button by default for safety
        if 'no_button' in locals():
            no_button.focus_set()
        
        # Center dialog on parent
        dialog.update_idletasks()  # Make sure dialog is updated for size calculations
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Handle dialog close via window manager
        dialog.protocol("WM_DELETE_WINDOW", on_no)
        
        # Wait for dialog to close
        parent.wait_window(dialog)
        
        return result[0]
    
    @staticmethod
    def handle_rejection(parent_dialog, image_path, metadata_callback=None, cleanup_callback=None):
        """
        Standardized handler for image rejection across all dialogs.
        
        This method:
        1. Validates that metadata has been entered
        2. Confirms the rejection with the user
        3. Collects a reason for rejection
        4. Returns metadata for the rejected image
        
        Note: This method does NOT handle file operations. The actual file movement
        and OneDrive upload is handled by the calling code using FileManager.save_original_file()
        
        Args:
            parent_dialog: The parent dialog window
            image_path: Path to the image being rejected
            metadata_callback: Callback to get/validate metadata, should return dict with:
                              - hole_id: str
                              - depth_from: int
                              - depth_to: int
                              Returns None if metadata is invalid
            cleanup_callback: Optional callback to clean up resources before closing
            
        Returns:
            dict: Result dictionary with:
                  - rejected: True
                  - hole_id, depth_from, depth_to: Metadata values
                  - rejection_reason: str - The reason for rejection
            None: If rejection was cancelled or failed
        """
        # Check thread safety
        DialogHelper._check_main_thread("handle_rejection")
        
        try:
            # Get and validate metadata
            if metadata_callback:
                metadata = metadata_callback()
                if not metadata:
                    # Metadata validation failed (callback should show its own error)
                    return None
            else:
                # No metadata callback provided - this is an error
                DialogHelper.show_message(
                    parent_dialog,
                    DialogHelper.t("Error"),
                    DialogHelper.t("Cannot reject image without metadata"),
                    message_type="error"
                )
                return None
            
            # Extract metadata
            hole_id = metadata.get('hole_id')
            depth_from = metadata.get('depth_from')
            depth_to = metadata.get('depth_to')
            
            # Final validation
            if not all([hole_id, depth_from is not None, depth_to is not None]):
                DialogHelper.show_message(
                    parent_dialog,
                    DialogHelper.t("Missing Information"),
                    DialogHelper.t("Please enter Hole ID and depth information before rejecting"),
                    message_type="error"
                )
                return None
            
            # Create custom rejection dialog with text entry
            rejection_dialog = tk.Toplevel(parent_dialog)
            rejection_dialog.title(DialogHelper.t("Confirm Rejection"))
            rejection_dialog.transient(parent_dialog)
            rejection_dialog.grab_set()
            
            # Center the dialog
            rejection_dialog.geometry("500x350")
            rejection_dialog.update_idletasks()
            x = (rejection_dialog.winfo_screenwidth() - rejection_dialog.winfo_width()) // 2
            y = (rejection_dialog.winfo_screenheight() - rejection_dialog.winfo_height()) // 2
            rejection_dialog.geometry(f"+{x}+{y}")
            
            # Variables
            rejection_reason = tk.StringVar(value="")
            dialog_result = {"confirmed": False, "reason": ""}
            
            # Create content
            main_frame = ttk.Frame(rejection_dialog, padding="20")
            main_frame.pack(fill='both', expand=True)
            
            # Info label
            info_text = DialogHelper.t(f"Are you sure you want to reject this image?\n\n"
                                     f"Hole ID: {hole_id}\n"
                                     f"Depth: {depth_from}-{depth_to}m\n\n"
                                     f"The image will be moved to the Failed and Skipped folder.")
            info_label = ttk.Label(main_frame, text=info_text, wraplength=450)
            info_label.pack(pady=(0, 15))
            
            # Reason frame
            reason_frame = ttk.LabelFrame(main_frame, 
                                        text=DialogHelper.t("Reason for Rejection"), 
                                        padding="10")
            reason_frame.pack(fill='both', expand=True, pady=(0, 15))
            
            # Reason text widget with scrollbar
            text_frame = ttk.Frame(reason_frame)
            text_frame.pack(fill='both', expand=True)
            
            reason_text = tk.Text(text_frame, height=4, width=50, wrap='word')
            reason_text.pack(side='left', fill='both', expand=True)
            
            scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=reason_text.yview)
            scrollbar.pack(side='right', fill='y')
            reason_text.configure(yscrollcommand=scrollbar.set)
            
            # Button frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x', pady=(10, 0))
            
            def on_reject():
                # Get the rejection reason
                reason = reason_text.get('1.0', 'end-1c').strip()
                if not reason:
                    DialogHelper.show_message(
                        rejection_dialog,
                        DialogHelper.t("Missing Reason"),
                        DialogHelper.t("Please provide a reason for rejection"),
                        message_type="warning"
                    )
                    return
                
                dialog_result["confirmed"] = True
                dialog_result["reason"] = reason
                rejection_dialog.destroy()
            
            def on_cancel():
                dialog_result["confirmed"] = False
                rejection_dialog.destroy()
            
            # Create buttons using ModernButton if available
            # Try multiple ways to find gui_manager
            gui_manager = None
            if hasattr(parent_dialog, 'gui_manager'):
                gui_manager = parent_dialog.gui_manager
            elif hasattr(parent_dialog, 'master') and hasattr(parent_dialog.master, 'gui_manager'):
                gui_manager = parent_dialog.master.gui_manager
            elif hasattr(parent_dialog, 'extractor') and hasattr(parent_dialog.extractor, 'gui_manager'):
                gui_manager = parent_dialog.extractor.gui_manager
            
            if gui_manager:
                # Apply theme to the dialog
                gui_manager.apply_theme(rejection_dialog)
                
                # Create modern buttons using theme colors
                reject_btn = gui_manager.create_modern_button(
                    button_frame,
                    text=DialogHelper.t("Reject"),
                    color=gui_manager.theme_colors["accent_red"],  # Use accent_red from theme
                    command=on_reject
                )
                cancel_btn = gui_manager.create_modern_button(
                    button_frame,
                    text=DialogHelper.t("Cancel"),
                    color=gui_manager.theme_colors["accent_green"],  # Use accent_green from theme
                    command=on_cancel
                )
                
                # Also apply theme to the text widget
                reason_text.configure(
                    bg=gui_manager.theme_colors["field_bg"],
                    fg=gui_manager.theme_colors["text"],
                    insertbackground=gui_manager.theme_colors["text"]
                )
            else:
                reject_btn = ttk.Button(button_frame, text=DialogHelper.t("Reject"), command=on_reject)
                cancel_btn = ttk.Button(button_frame, text=DialogHelper.t("Cancel"), command=on_cancel)
            
            cancel_btn.pack(side='right', padx=(5, 0))
            reject_btn.pack(side='right')
            
            # Bind Enter key to reject (with reason check)
            rejection_dialog.bind('<Return>', lambda e: on_reject())
            rejection_dialog.bind('<Escape>', lambda e: on_cancel())
            
            # Focus on text entry
            reason_text.focus_set()
            
            # Wait for dialog to close
            rejection_dialog.wait_window()
            
            # Check result
            if not dialog_result["confirmed"]:
                return None  # User cancelled
            
            # Clean up resources if callback provided
            if cleanup_callback:
                cleanup_callback()
            
            # Return rejection result with reason
            return {
                'rejected': True,
                'hole_id': hole_id,
                'depth_from': int(depth_from),
                'depth_to': int(depth_to),
                'compartment_interval': metadata.get('compartment_interval', 1),
                'rejection_reason': dialog_result["reason"]
            }
            
        except Exception as e:
            logger.error(f"Error in handle_rejection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            DialogHelper.show_message(
                parent_dialog,
                DialogHelper.t("Error"),
                DialogHelper.t(f"An error occurred during rejection: {str(e)}"),
                message_type="error"
            )
            return None

    @staticmethod
    def center_dialog(dialog, size_ratio=0.8, min_width=400, min_height=300, max_width=None, max_height=None):
        """
        Center a dialog on screen and set appropriate size.
        
        Args:
            dialog: Dialog window to center
            size_ratio: Ratio of screen size to use (0.0-1.0) or None to use only min/max constraints
            min_width: Minimum width in pixels
            min_height: Minimum height in pixels
            max_width: Maximum width in pixels (optional)
            max_height: Maximum height in pixels (optional)
        """
        # Check thread safety
        DialogHelper._check_main_thread("center_dialog")
        
        # Get screen dimensions
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        # Calculate size based on ratio but ensure minimum size
        if size_ratio is not None:
            width = max(min_width, int(screen_width * size_ratio))
            height = max(min_height, int(screen_height * size_ratio))
        else:
            # If no ratio specified, just use the minimum sizes
            width = min_width
            height = min_height
        
        # Apply maximum constraints if specified
        if max_width:
            width = min(width, max_width)
        if max_height:
            height = min(height, max_height)
        
        # Make sure size doesn't exceed screen
        width = min(width, screen_width - 100)
        height = min(height, screen_height - 100)
        
        # Calculate position to center dialog
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set geometry
        dialog.geometry(f"{width}x{height}+{x}+{y}")