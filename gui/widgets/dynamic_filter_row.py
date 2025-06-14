# gui/widgets/dynamic_filter_row.py

import tkinter as tk
from tkinter import ttk
import pandas as pd
from gui.dialog_helper import DialogHelper

class DynamicFilterRow:
    """A single filter row with column selection, operator, and value input."""
    
    def __init__(self, parent, gui_manager, columns_info, register_data, on_remove_callback, index):
        """
        Initialize a filter row.
        
        Args:
            parent: Parent widget
            gui_manager: GUIManager instance for theming
            columns_info: Dict mapping column names to their data types
            register_data: The pandas DataFrame with all data
            on_remove_callback: Function to call when removing this row
            index: Row index
        """
        self.parent = parent
        self.gui_manager = gui_manager
        self.columns_info = columns_info
        self.register_data = register_data  # Store the data reference
        self.on_remove_callback = on_remove_callback
        self.index = index
        
        # Variables
        self.column_var = tk.StringVar()
        self.operator_var = tk.StringVar()
        self.value_var = tk.StringVar()
        self.value2_var = tk.StringVar()  # For 'between' operator
        self.selected_values = []  # For multi-select
        
        # Create the row frame
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, pady=2)
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the widgets for this filter row."""
        # Column selector with themed dropdown
        columns = list(self.columns_info.keys())
        
        # Create custom styled dropdown frame for column
        column_dropdown_frame = tk.Frame(
            self.frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        column_dropdown_frame.pack(side=tk.LEFT, padx=(0, 5))
        
        self.column_dropdown = tk.OptionMenu(
            column_dropdown_frame,
            self.column_var,
            "",  # Initial value
            *columns
        )
        self.gui_manager.style_dropdown(self.column_dropdown, width=15)
        self.column_dropdown.pack()
        
        # Set initial column if available
        if columns:
            self.column_var.set(columns[0])
        
        # Trace column changes
        self.column_var.trace_add('write', self._on_column_change)
        
        # Operator selector with themed dropdown
        operator_dropdown_frame = tk.Frame(
            self.frame,
            bg=self.gui_manager.theme_colors["field_bg"],
            highlightbackground=self.gui_manager.theme_colors["field_border"],
            highlightthickness=1,
            bd=0
        )
        operator_dropdown_frame.pack(side=tk.LEFT, padx=(0, 5))
        
        self.operator_dropdown = tk.OptionMenu(
            operator_dropdown_frame,
            self.operator_var,
            ""  # Initial value will be set by column change
        )
        self.gui_manager.style_dropdown(self.operator_dropdown, width=12)
        self.operator_dropdown.pack()
        
        # Trace operator changes
        self.operator_var.trace_add('write', self._on_operator_change)
        
        # Value input container
        self.value_frame = ttk.Frame(self.frame)
        self.value_frame.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        # Remove button
        remove_btn = tk.Button(
            self.frame,
            text="✕",
            command=lambda: self.on_remove_callback(self.index),
            bg=self.gui_manager.theme_colors["accent_red"],
            fg="white",
            font=("Arial", 10, "bold"),
            bd=0,
            padx=5,
            pady=2,
            cursor="hand2"
        )
        remove_btn.pack(side=tk.RIGHT, padx=(5, 0))

    def _on_column_change(self, *args):
        """Handle column selection change."""
        column = self.column_var.get()
        if not column:
            return
            
        # Clear value frame
        for widget in self.value_frame.winfo_children():
            widget.destroy()
            
        # Update operators based on column type
        col_type = self.columns_info.get(column, 'text')
        
        if col_type == 'numeric':
            operators = ['<', '<=', '=', '>=', '>', '!=', 'between']
        else:
            operators = ['is', 'is not', 'contains', 'starts with', 'ends with', 'in']
            
        # Update the operator dropdown menu
        menu = self.operator_dropdown['menu']
        menu.delete(0, 'end')
        
        for op in operators:
            menu.add_command(label=op, command=tk._setit(self.operator_var, op))
            
        if operators:
            self.operator_var.set(operators[0])

    def _on_operator_change(self, *args):
        """Handle operator selection change."""
        column = self.column_var.get()
        operator = self.operator_var.get()
        
        if not column or not operator:
            return
            
        # Clear value frame
        for widget in self.value_frame.winfo_children():
            widget.destroy()
            
        col_type = self.columns_info.get(column, 'text')
        
        if col_type == 'numeric':
            self._create_numeric_value_input(operator)
        else:
            self._create_text_value_input(operator)




    def _create_numeric_value_input(self, operator):
        """Create value input for numeric columns."""
        # Create validation function that gets called on every change
        def validate_and_update(*args):
            self._validate_numeric()
        
        if operator == 'between':
            # Create two inputs for between
            self.value_entry = self.gui_manager.create_entry_with_validation(
                self.value_frame,
                self.value_var,
                validate_func=validate_and_update,
                width=10,
                placeholder="min"
            )
            self.value_entry.pack(side=tk.LEFT, padx=(0, 2))
            
            ttk.Label(
                self.value_frame,
                text="-",
                style='Content.TLabel'
            ).pack(side=tk.LEFT, padx=2)
            
            # Create separate validation for second value
            def validate_value2(*args):
                value = self.value2_var.get()
                if not value or value in ["-", ".", "-."]:
                    return
                    
                try:
                    float(value)
                except ValueError:
                    cleaned = ""
                    has_decimal = False
                    has_minus = False
                    
                    for i, char in enumerate(value):
                        if char == '-' and i == 0 and not has_minus:
                            cleaned += char
                            has_minus = True
                        elif char == '.' and not has_decimal:
                            cleaned += char
                            has_decimal = True
                        elif char.isdigit():
                            cleaned += char
                    
                    self.value2_var.set(cleaned)
            
            self.value2_entry = self.gui_manager.create_entry_with_validation(
                self.value_frame,
                self.value2_var,
                validate_func=validate_value2,
                width=10,
                placeholder="max"
            )
            self.value2_entry.pack(side=tk.LEFT, padx=(2, 0))
        else:
            # Single input for other operators
            self.value_entry = self.gui_manager.create_entry_with_validation(
                self.value_frame,
                self.value_var,
                validate_func=validate_and_update,
                width=20
            )
            self.value_entry.pack(side=tk.LEFT)

    def _create_text_value_input(self, operator):
        """Create value input for text columns."""
        column = self.column_var.get()
        
        if operator in ['is', 'is not']:
            # Single select dropdown with proper theming
            unique_values = self._get_unique_values(column)
            
            # If no unique values, show a message
            if not unique_values:
                ttk.Label(
                    self.value_frame,
                    text=DialogHelper.t("No values available"),
                    style='Content.TLabel'
                ).pack(side=tk.LEFT)
                return
            
            # Create custom styled dropdown frame
            dropdown_frame = tk.Frame(
                self.value_frame,
                bg=self.gui_manager.theme_colors["field_bg"],
                highlightbackground=self.gui_manager.theme_colors["field_border"],
                highlightthickness=1,
                bd=0
            )
            dropdown_frame.pack(side=tk.LEFT)
            
            # Create the OptionMenu widget
            self.value_dropdown = tk.OptionMenu(
                dropdown_frame,
                self.value_var,
                unique_values[0],  # Default value
                *unique_values
            )
            
            # Style the dropdown
            self.gui_manager.style_dropdown(self.value_dropdown, width=20)
            self.value_dropdown.pack()
            
            # Set initial value
            self.value_var.set(unique_values[0])
            
        elif operator == 'in':
            # For 'in' operator, show available values as a hint
            unique_values = self._get_unique_values(column)
            hint_text = f"Available: {', '.join(unique_values[:5])}" if unique_values else ""
            if len(unique_values) > 5:
                hint_text += "..."
                
            # Multi-select (simplified - using entry for comma-separated values)
            ttk.Label(
                self.value_frame,
                text=DialogHelper.t("Values (comma-separated):"),
                style='Content.TLabel'
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            self.value_entry = self.gui_manager.create_entry_with_validation(
                self.value_frame,
                self.value_var,
                width=30,
                placeholder="value1, value2, ..."
            )
            self.value_entry.pack(side=tk.LEFT)
            
            if hint_text:
                ttk.Label(
                    self.value_frame,
                    text=hint_text,
                    style='Content.TLabel',
                    font=('Arial', 8)
                ).pack(side=tk.LEFT, padx=(5, 0))
            
        else:
            # Text input for contains, starts with, ends with
            self.value_entry = self.gui_manager.create_entry_with_validation(
                self.value_frame,
                self.value_var,
                width=25
            )
            self.value_entry.pack(side=tk.LEFT)
            
    def _validate_numeric(self, *args):
        """Validate numeric input."""
        value = self.value_var.get()
        
        # Allow empty input
        if not value:
            return
            
        # Allow just a minus sign or decimal point at the beginning
        if value in ["-", ".", "-."]:
            return
            
        try:
            # Try to convert to float
            float(value)
        except ValueError:
            # Remove non-numeric characters, keeping only digits, minus, and decimal point
            cleaned = ""
            has_decimal = False
            has_minus = False
            
            for i, char in enumerate(value):
                if char == '-' and i == 0 and not has_minus:
                    cleaned += char
                    has_minus = True
                elif char == '.' and not has_decimal:
                    cleaned += char
                    has_decimal = True
                elif char.isdigit():
                    cleaned += char
            
            # Set the cleaned value
            self.value_var.set(cleaned)


    def _get_unique_values(self, column):
        """Get unique values for a column."""
        try:
            if self.register_data is not None and column in self.register_data.columns:
                unique_vals = self.register_data[column].dropna().unique().tolist()
                # Convert to strings and sort
                return sorted([str(v) for v in unique_vals])
        except Exception as e:
            import logging
            logging.error(f"Error getting unique values for column {column}: {e}")
        
        return []  # Return empty list instead of ['All']
        
    def get_filter_config(self):
        """Get the current filter configuration."""
        return {
            'column': self.column_var.get(),
            'operator': self.operator_var.get(),
            'value': self.value_var.get(),
            'value2': self.value2_var.get() if self.operator_var.get() == 'between' else None
        }
        
    def destroy(self):
        """Destroy this filter row."""
        self.frame.destroy()