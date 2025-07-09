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
    def t(cls, text, *args, **kwargs):
        """
        Translate text using the class translator.
        
        Args:
            text (str): The text key to translate
            *args: Positional arguments for string formatting
            **kwargs: Keyword arguments for string formatting
            
        Returns:
            str: The translated text or original text if no translation available
        """
        if cls.translator:
            # Pass through all arguments to the translator
            return cls.translator.translate(text, *args, **kwargs)
        else:
            # No translator available - try basic formatting
            try:
                if args or kwargs:
                    return text.format(*args, **kwargs)
                return text
            except:
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
    def create_dialog(parent, title, modal=True, topmost=True):
        """
        Create a properly configured dialog window with theme styling.
        This ONLY creates the dialog - it does NOT position or size it.
        
        Args:
            parent: Parent window
            title: Dialog title
            modal: Whether dialog is modal
            topmost: Whether dialog stays on top
            
        Returns:
            Configured dialog window (not positioned)
        """
        logger.debug(f"DialogHelper.create_dialog called with title: {title}")
        
        # Check thread safety
        DialogHelper._check_main_thread("create_dialog")
        
        dialog = tk.Toplevel(parent)
        dialog.title(DialogHelper.t(title))
        
        # Get DPI scaling factor
        try:
            scaling_factor = dialog.tk.call('tk', 'scaling')
            if scaling_factor > 0:
                dpi_scale = scaling_factor / 1.333333  # Convert to percentage
            else:
                dpi_scale = 1.0
        except:
            dpi_scale = 1.0
        
        logger.debug(f"DPI scaling factor: {dpi_scale:.2f}")
        
        # Store DPI scale in dialog for child widgets to use
        dialog.dpi_scale = dpi_scale
        
        # Apply theme colors if gui_manager is available
        if DialogHelper.gui_manager and hasattr(DialogHelper.gui_manager, 'theme_colors'):
            theme_colors = DialogHelper.gui_manager.theme_colors
            dialog.configure(bg=theme_colors["background"])
        
        # Set window behavior
        if parent:
            dialog.transient(parent)  # Make dialog transient to parent
        
        if modal:
            dialog.grab_set()  # Make dialog modal

        
        if topmost:
            dialog.attributes('-topmost', True)  # Keep on top
        
        # Make sure dialog is ready but don't position it
        dialog.update_idletasks()
        
        return dialog

    @staticmethod
    def center_dialog(dialog, parent=None, size_ratio=None, 
                    max_width=None, max_height=None,
                    min_width=None, min_height=None):  # Add these parameters
        """
        Center a dialog window using its natural content size.
        Call this AFTER all content has been added to the dialog.
        
        Args:
            dialog: Dialog window to center
            parent: Parent window for relative positioning (None = center on screen)
            size_ratio: Optional ratio to scale the natural size (e.g., 1.2 = 120% of content)
            max_width: Maximum width constraint (optional)
            max_height: Maximum height constraint (optional)
            min_width: Minimum width constraint (optional)
            min_height: Minimum height constraint (optional)
        """
        # Check thread safety
        DialogHelper._check_main_thread("center_dialog")
        
        # Ensure dialog is updated to get accurate sizes
        dialog.update_idletasks()
        
        # Get screen dimensions
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        # Get the dialog's natural content size (what it needs to fit everything)
        natural_width = dialog.winfo_reqwidth()
        natural_height = dialog.winfo_reqheight()
        
        logger.debug(f"Dialog natural size: {natural_width}x{natural_height}")
        
        # Start with the natural size
        width = natural_width
        height = natural_height
        
        # Apply optional scaling ratio if provided
        if size_ratio is not None and size_ratio > 0:
            width = int(natural_width * size_ratio)
            height = int(natural_height * size_ratio)
            logger.debug(f"After ratio {size_ratio}: {width}x{height}")
        
        # Apply minimum constraints if specified
        if min_width and width < min_width:
            width = min_width
        if min_height and height < min_height:
            height = min_height
        
        # Apply maximum constraints if specified
        if max_width and width > max_width:
            width = max_width
        if max_height and height > max_height:
            height = max_height
        
        # Ensure doesn't exceed screen bounds (with margin for taskbar)
        screen_margin = 50  # Margin from screen edges
        taskbar_margin = 100  # Extra bottom margin for taskbar
        
        width = min(width, screen_width - (2 * screen_margin))
        height = min(height, screen_height - taskbar_margin - screen_margin)
        
        # Calculate center position
        if parent and parent.winfo_exists() and parent.winfo_viewable():
            # Center on parent
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()
            
            x = parent_x + (parent_width - width) // 2
            y = parent_y + (parent_height - height) // 2
        else:
            # Center on screen
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
        
        # Ensure dialog stays on screen
        x = max(screen_margin, min(x, screen_width - width - screen_margin))
        y = max(screen_margin, min(y, screen_height - height - taskbar_margin))
        
        # Apply geometry
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Ensure dialog is visible
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()
        
        logger.debug(f"Dialog positioned: {width}x{height}+{x}+{y}")

    @staticmethod
    def _position_dialog_on_parent(dialog, parent):
        """
        Helper method to position a dialog centered on its parent.
        Used by show_message, confirm_dialog, etc.
        """
        dialog.update_idletasks()  # Ensure size is calculated
        
        if parent and parent.winfo_exists() and parent.winfo_viewable():
            # Center on parent
            dialog_width = dialog.winfo_width()
            dialog_height = dialog.winfo_height()
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()
            
            x = parent_x + (parent_width - dialog_width) // 2
            y = parent_y + (parent_height - dialog_height) // 2
            
            # Ensure not off-screen
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = max(0, min(x, screen_width - dialog_width))
            y = max(0, min(y, screen_height - dialog_height))
            
            dialog.geometry(f"+{x}+{y}")
        else:
            # Center on screen
            dialog_width = dialog.winfo_width()
            dialog_height = dialog.winfo_height()
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            
            x = (screen_width - dialog_width) // 2
            y = (screen_height - dialog_height) // 2
            
            dialog.geometry(f"+{x}+{y}")
        
        # Ensure visibility
        dialog.lift()
        dialog.focus_force()

    @staticmethod
    def create_error_dialog(parent, title, message, **kwargs):
        """
        Create a properly configured error dialog with themed components.
        
        Args:
            parent: Parent window
            title: Dialog title (will be translated)
            message: Error message (will be translated)
            **kwargs: Additional keyword arguments for message translation
        """
        # Check thread safety
        DialogHelper._check_main_thread("create_error_dialog")
        
        # Translate title and message
        title = DialogHelper.t(title)
        message = DialogHelper.t(message, **kwargs)
        
        # Create base dialog
        dialog = DialogHelper.create_dialog(
            parent, 
            title,  # Already translated
            modal=True, 
            topmost=True
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
            ok_button = DialogHelper.gui_manager.create_modern_button(
                dialog,
                text=DialogHelper.t("OK"),
                color=theme_colors["accent_blue"],
                command=dialog.destroy
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

        # Center with constraints after content is added
        dialog.update_idletasks()
        DialogHelper.center_dialog(
            dialog,
            parent,
            size_ratio=0.3,  # Smaller ratio for error dialogs
            min_width=350,
            min_height=180
        )
        
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
            parent, title, modal=True, topmost=True
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
            ok_button = DialogHelper.gui_manager.create_modern_button(
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

            # Center with constraints after content is added
        dialog.update_idletasks()
        DialogHelper.center_dialog(
            dialog,
            parent,
            size_ratio=0.4,
            min_width=300,
            min_height=150
        )
        
        return dialog
    

    @staticmethod
    def show_message(parent, title, message, message_type="info", **kwargs):
        """
        Show a message dialog with an OK button.
        
        Args:
            parent: Parent window
            title: Dialog title (will be translated)
            message: Message to display (will be translated)
            message_type: Type of message ("info", "error", "warning", "success")
            **kwargs: Additional keyword arguments for message translation
        """
        # Check thread safety
        DialogHelper._check_main_thread("show_message")
        
        # Translate title and message
        title = DialogHelper.t(title)
        message = DialogHelper.t(message, **kwargs)
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.transient(parent)
        dialog.grab_set()
        
        # Apply theme if available
        if DialogHelper.gui_manager:
            DialogHelper.gui_manager.apply_theme(dialog)
        
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
        msg_label = ttk.Label(
            top_frame, 
            text=message, 
            wraplength=400,
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
        
        # ===================================================
        # SIMPLIFIED: Center on parent or screen
        # ===================================================
        def center_message_dialog():
            dialog.update_idletasks()
            
            # Get dialog size
            width = dialog.winfo_reqwidth()
            height = dialog.winfo_reqheight()
            
            # Get screen dimensions
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            
            if parent and parent.winfo_viewable():
                # Center on parent
                parent_x = parent.winfo_rootx()
                parent_y = parent.winfo_rooty()
                parent_width = parent.winfo_width()
                parent_height = parent.winfo_height()
                
                x = parent_x + (parent_width - width) // 2
                y = parent_y + (parent_height - height) // 2
            else:
                # Center on screen
                x = (screen_width - width) // 2
                y = (screen_height - height) // 2
            
            # Ensure not off-screen
            x = max(0, min(x, screen_width - width))
            y = max(0, min(y, screen_height - height))
            
            dialog.geometry(f"{width}x{height}+{x}+{y}")
            dialog.lift()
            dialog.focus_force()
        
        # Position after layout is complete
        dialog.after_idle(center_message_dialog)
        
        # Handle dialog close via window manager
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        
        # Wait for dialog to close
        parent.wait_window(dialog)

    @staticmethod
    def confirm_dialog(parent, title, message, yes_text="Yes", no_text="No", **kwargs):
        """
        Show a confirmation dialog with Yes/No buttons.
        
        Args:
            parent: Parent window
            title: Dialog title (will be translated)
            message: Message to display (will be translated)
            yes_text: Text for Yes button (will be translated)
            no_text: Text for No button (will be translated)
            **kwargs: Additional keyword arguments for message translation
        """
        # Check thread safety
        DialogHelper._check_main_thread("confirm_dialog")
        
        # Translate all text
        title = DialogHelper.t(title)
        message = DialogHelper.t(message, **kwargs)
        yes_text = DialogHelper.t(yes_text)
        no_text = DialogHelper.t(no_text)

        # Check thread safety
        DialogHelper._check_main_thread("confirm_dialog")
        
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.transient(parent)
        dialog.grab_set()
        
        # Apply theme if available
        if DialogHelper.gui_manager:
            DialogHelper.gui_manager.apply_theme(dialog)
        
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
        no_button.focus_set()
        
        # Position after layout is complete
        dialog.after_idle(lambda: DialogHelper._position_dialog_on_parent(dialog, parent))
        
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
            
            # Create custom rejection dialog
            rejection_dialog = DialogHelper.create_dialog(
                parent_dialog,
                "Confirm Rejection",
                modal=True,
                topmost=True
            )
            
            # Variables
            rejection_reason = tk.StringVar(value="")
            dialog_result = {"confirmed": False, "reason": ""}
            
            # Apply theme if available
            theme_colors = None
            gui_manager = DialogHelper.gui_manager
            if gui_manager and hasattr(gui_manager, 'theme_colors'):
                theme_colors = gui_manager.theme_colors
                gui_manager.apply_theme(rejection_dialog)
            
            # Create content
            main_frame = ttk.Frame(rejection_dialog, padding="20")
            main_frame.pack(fill='both', expand=True)
            
            # Info label
            info_text = f"Are you sure you want to reject this image?\n\n" \
                       f"Hole ID: {hole_id}\n" \
                       f"Depth: {depth_from}-{depth_to}m\n\n" \
                       f"The image will be moved to the Failed and Skipped folder."
            info_label = ttk.Label(main_frame, text=info_text, wraplength=450, justify=tk.CENTER)
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
            
            # Apply theme to text widget
            if theme_colors:
                reason_text.configure(
                    bg=theme_colors["field_bg"],
                    fg=theme_colors["text"],
                    insertbackground=theme_colors["text"],
                    relief=tk.FLAT,
                    highlightbackground=theme_colors["field_border"],
                    highlightthickness=1
                )
            
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
            if gui_manager and theme_colors:
                reject_btn = gui_manager.create_modern_button(
                    button_frame,
                    text=DialogHelper.t("Reject"),
                    color=theme_colors["accent_red"],
                    command=on_reject
                )
                cancel_btn = gui_manager.create_modern_button(
                    button_frame,
                    text=DialogHelper.t("Cancel"),
                    color=theme_colors["accent_green"],
                    command=on_cancel
                )
            else:
                reject_btn = ttk.Button(button_frame, text=DialogHelper.t("Reject"), command=on_reject)
                cancel_btn = ttk.Button(button_frame, text=DialogHelper.t("Cancel"), command=on_cancel)
            
            cancel_btn.pack(side='right', padx=(5, 0))
            reject_btn.pack(side='right')
            
            # Bind Enter key to reject (with reason check)
            rejection_dialog.bind('<Return>', lambda e: on_reject())
            rejection_dialog.bind('<Escape>', lambda e: on_cancel())
            
            # Then center it with size constraints after content is added
            rejection_dialog.update_idletasks()
            DialogHelper.center_dialog(
                rejection_dialog,
                parent_dialog,
                size_ratio=0.5,
                max_width= 800,
                max_height= 800
            )
            
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