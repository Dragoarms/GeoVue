import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime, date as date_type
from gui.dialog_helper import DialogHelper
from gui.widgets.modern_button import ModernButton

# Optional calendar widget for date filters
try:
    from gui.widgets.themed_date_entry import create_themed_date_entry
    _HAS_DATE_ENTRY = True
except ImportError:
    _HAS_DATE_ENTRY = False


class DynamicFilterRow:
    """Enhanced filter row with column selection, data type detection, and appropriate operators."""

    # Define operators for each data type
    NUMERIC_OPERATORS = {
        "=": "equals",
        "≠": "not equals",
        "<": "less than",
        "≤": "less than or equal",
        ">": "greater than",
        "≥": "greater than or equal",
        "between": "between",
        "is null": "is null",
        "not null": "not null",
    }

    TEXT_OPERATORS = {
        "equals": "equals",
        "not equals": "not equals",
        "contains": "contains",
        "not contains": "not contains",
        "starts with": "starts with",
        "ends with": "ends with",
        "in": "in list",
        "not in": "not in list",
        "like": "pattern match",
        "is null": "is null",
        "not null": "not null",
    }

    DATE_OPERATORS = {
        "on": "on date",
        "not on": "not on date",
        "before": "before date",
        "on or before": "on or before",
        "after": "after date",
        "on or after": "on or after",
        "between": "between dates",
        "is null": "is null",
        "not null": "not null",
    }

    # Common date formats to try when parsing date values
    DATE_PARSE_FORMATS = [
        "%Y-%m-%d",           # 2024-01-15
        "%Y-%m-%d %H:%M:%S",  # 2024-01-15 10:30:00
        "%d/%m/%Y",           # 15/01/2024
        "%d-%m-%Y",           # 15-01-2024
        "%m/%d/%Y",           # 01/15/2024
        "%Y%m%d",             # 20240115
        "%d %b %Y",           # 15 Jan 2024
        "%d %B %Y",           # 15 January 2024
    ]

    def __init__(
        self,
        parent,
        gui_manager,
        columns_info,
        register_data,
        on_remove_callback,
        index,
        on_column_selected_callback=None,
        get_column_schema_callback=None,
        get_unique_values_callback=None,
    ):
        """Initialize an enhanced filter row."""
        self.parent = parent
        self.gui_manager = gui_manager
        self.columns_info = columns_info if columns_info else {}
        self.register_data = register_data
        self.on_remove_callback = on_remove_callback
        self.index = index
        self.on_column_selected_callback = on_column_selected_callback
        self.get_column_schema_callback = get_column_schema_callback
        self.get_unique_values_callback = get_unique_values_callback

        # Theme colors - use both names for compatibility
        self.theme = gui_manager.theme_colors
        self.theme_colours = gui_manager.theme_colors  # British spelling alias

        # Debug: Verify theme colors
        print(f"DEBUG DynamicFilterRow: Theme colors received:")
        print(f"  field_bg: {self.theme.get('field_bg', 'MISSING')}")
        print(f"  text: {self.theme.get('text', 'MISSING')}")
        print(f"  field_border: {self.theme.get('field_border', 'MISSING')}")
        self.fonts = {"normal": ("Arial", 10), "small": ("Arial", 9)}

        # Variables
        self.column_var = tk.StringVar()
        self.data_type_var = tk.StringVar(value="auto")
        self.operator_var = tk.StringVar()
        self.value_var = tk.StringVar()
        self.value2_var = tk.StringVar()  # For 'between' operator

        # State tracking
        self.current_data_type = None
        self.unique_values_cache = {}

        # Create the row frame with proper styling
        self.frame = tk.Frame(parent, bg=self.theme_colours["background"])
        self.frame.pack(fill=tk.X, pady=2)
        # Let it size naturally based on content

        # Debug visibility
        print(f"DynamicFilterRow frame created: parent={parent}, frame={self.frame}")

        self._create_widgets()

    def _create_widgets(self):
        """Create the enhanced filter row widgets."""
        # 1. Column selector
        # Get columns from register_data if columns_info is empty
        if not self.columns_info and self.register_data is not None:
            columns = list(self.register_data.columns)
        else:
            columns = list(self.columns_info.keys())

        print(
            f"DEBUG: Available columns: {columns[:10]}..."
        )  # Show first 10 for debugging

        # Row 1 frame
        self.row1_frame = tk.Frame(self.frame, bg=self.theme_colours["background"])
        self.row1_frame.pack(fill=tk.X, pady=(0, 2))

        # Column dropdown with frame for border
        column_frame = tk.Frame(
            self.row1_frame,
            bg=self.theme_colours["field_bg"],
            highlightbackground=self.theme_colours["field_border"],
            highlightthickness=1,
            bd=0,
        )
        column_frame.pack(side=tk.LEFT, padx=(0, 2))

        if columns:
            self.column_var.set(columns[0])
        
        # Use themed searchable optionmenu for column selection
        try:
            self.column_dropdown = self.gui_manager.create_searchable_optionmenu(
                parent=column_frame,
                items=columns if columns else [""],
                variable=self.column_var,
                width=15,
                placeholder="Select column...",
                on_change=None,  # We use trace_add below instead
            )
            self.column_dropdown.pack()
        except Exception as e:
            # Fallback to standard dropdown
            print(f"Failed to create searchable dropdown, using standard: {e}")
            if columns:
                self.column_dropdown = tk.OptionMenu(
                    column_frame, self.column_var, *columns
                )
            else:
                self.column_dropdown = tk.OptionMenu(column_frame, self.column_var, "")
            self.gui_manager.style_dropdown(self.column_dropdown, width=15)
            self.column_dropdown.pack()

        # 2. Data type selector (auto/numeric/text)
        type_frame = tk.Frame(
            self.row1_frame,
            bg=self.theme_colours["field_bg"],
            highlightbackground=self.theme_colours["field_border"],
            highlightthickness=1,
            bd=0,
        )
        type_frame.pack(side=tk.LEFT, padx=(0, 2))

        type_options = ["auto", "numeric", "text", "date"]
        self.type_dropdown = tk.OptionMenu(
            type_frame, self.data_type_var, *type_options
        )
        self.gui_manager.style_dropdown(self.type_dropdown, width=7)
        self.type_dropdown.pack()

        # Row 2 frame for operator and value
        self.row2_frame = tk.Frame(self.frame, bg=self.theme_colours["background"])
        self.row2_frame.pack(fill=tk.X, pady=(0, 2))

        # 3. Operator selector (will be populated based on data type)
        operator_frame = tk.Frame(
            self.row2_frame,
            bg=self.theme_colours["field_bg"],
            highlightbackground=self.theme_colours["field_border"],
            highlightthickness=1,
            bd=0,
        )
        operator_frame.pack(side=tk.LEFT, padx=(0, 2))

        # Initialize with empty option
        self.operator_dropdown = tk.OptionMenu(operator_frame, self.operator_var, "")
        self.gui_manager.style_dropdown(self.operator_dropdown, width=12)
        self.operator_dropdown.pack()

        # 4. Value input container
        self.value_frame = tk.Frame(
            self.row2_frame, bg=self.theme_colours["background"]
        )
        self.value_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        # 5. Remove button (on row 1)
        self.remove_btn = ModernButton(
            self.row1_frame,
            text="✖",
            command=lambda: self.on_remove_callback(self.index),
            color=self.theme_colours["accent_red"],
            theme_colors=self.theme,
        )
        self.remove_btn.pack(side=tk.RIGHT, padx=(2, 0))

        # Set up traces for dynamic updates
        self.column_var.trace_add("write", self._on_column_change)
        self.data_type_var.trace_add("write", self._on_data_type_change)
        self.operator_var.trace_add("write", self._on_operator_change)

        # Initialize with first column if available
        if columns:
            self._on_column_change()

        # Debug - check if widgets are visible
        self.frame.update_idletasks()
        print(
            f"Filter row created - width: {self.frame.winfo_width()}, height: {self.frame.winfo_height()}"
        )
        print(f"Filter row parent: {self.frame.master}")

    def _detect_data_type(self, column):
        """Auto-detect data type for a column, handling nulls/gaps."""
        if not column:
            return "text"

        # Handle column not in register_data (lazy-loaded columns etc.)
        col_base = column.split(" (")[0].strip()
        col_found = None
        if self.register_data is not None:
            for candidate in [column, col_base, col_base.lower(), col_base.upper()]:
                if candidate in self.register_data.columns:
                    col_found = candidate
                    break

        if col_found is None:
            # No data to sniff — use column name heuristics
            return self._detect_type_from_name(col_base)

        col_data = self.register_data[col_found]

        # Remove nulls for analysis
        non_null_data = col_data.dropna()

        if len(non_null_data) == 0:
            return self._detect_type_from_name(col_base)

        # 1. Check if dtype is already datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return "date"

        # 2. Check if actual python datetime/date objects
        sample = non_null_data.head(20)
        datetime_obj_count = sum(
            1 for v in sample
            if isinstance(v, (datetime, date_type, pd.Timestamp))
        )
        if datetime_obj_count > len(sample) * 0.8:
            return "date"

        # 3. Check if numeric
        try:
            numeric_data = pd.to_numeric(non_null_data, errors="coerce")
            success_rate = numeric_data.notna().sum() / len(non_null_data)
            if success_rate > 0.8:
                return "numeric"
        except Exception:
            pass

        # 4. Check if string values parse as dates (only for non-numeric strings)
        if self._detect_type_from_name(col_base) == "date":
            # Column name suggests date — try parsing a sample
            if self._sample_parses_as_date(non_null_data.head(30)):
                return "date"

        # 5. Even without a name hint, try date parsing if values look date-ish
        str_sample = non_null_data.head(20).astype(str)
        # Quick heuristic: if most values contain '-' or '/' and have 8+ chars, try parsing
        separators = str_sample.str.contains(r'[-/]', regex=True, na=False)
        if separators.sum() > len(str_sample) * 0.7:
            if self._sample_parses_as_date(non_null_data.head(30)):
                return "date"

        return "text"

    def _detect_type_from_name(self, column_name: str) -> str:
        """Guess data type from column name alone."""
        name_lower = column_name.lower().replace('_', '').replace(' ', '')
        date_hints = [
            'date', 'time', 'created', 'modified', 'updated', 'logged',
            'startdate', 'enddate', 'loggeddate', 'sampledate',
            'approvaldate', 'rehabdate', 'cleareddate', 'completedate',
            'duedate', 'expirydate', 'returndate',
        ]
        for hint in date_hints:
            if hint in name_lower:
                return "date"
        return "text"

    def _sample_parses_as_date(self, sample_data) -> bool:
        """Check if a sample of values can be parsed as dates."""
        try:
            str_values = sample_data.astype(str)
            # Filter out obvious non-dates
            str_values = str_values[~str_values.isin(['', 'None', 'nan', 'null', 'NaN'])]
            if len(str_values) == 0:
                return False
            parsed = pd.to_datetime(str_values, errors='coerce', infer_datetime_format=True)
            success_rate = parsed.notna().sum() / len(str_values)
            return success_rate > 0.7
        except Exception:
            return False

    def _on_column_change(self, *args):
        """Handle column selection change."""
        column = self.column_var.get()
        if not column:
            return

        # Notify parent to load data for this column if needed (for lazy-loaded CSV columns)
        if self.on_column_selected_callback:
            self.on_column_selected_callback(column)

        # Always re-detect data type when column changes
        detected_type = self._detect_data_type(column)
        self.current_data_type = detected_type
        # Update the dropdown display (temporarily disable trace to avoid recursion)
        traces = self.data_type_var.trace_info()
        if traces:
            trace_id = traces[0][1]
            self.data_type_var.trace_remove("write", trace_id)
        self.data_type_var.set(detected_type)
        self.data_type_var.trace_add("write", self._on_data_type_change)

        # Update operators based on data type
        self._update_operators()

        # Cache unique values for this column (data should be loaded by callback now)
        self._cache_unique_values(column)  # Always update cache, not just if missing

        # Recreate the value input to use new unique values
        # Only if we have an operator selected (i.e., value input is shown)
        if self.operator_var.get():
            self._on_operator_change()  # This will recreate the value input

    def _on_data_type_change(self, *args):
        """Handle manual data type change."""
        data_type = self.data_type_var.get()

        if data_type == "auto":
            # Re-detect based on current column
            column = self.column_var.get()
            if column:
                detected_type = self._detect_data_type(column)
                self.current_data_type = detected_type
                # Update visual display to show detected type
                if data_type == "auto":  # Still auto after the get
                    # Temporarily disable trace to avoid recursion
                    trace_id = self.data_type_var.trace_info()[0][1]
                    self.data_type_var.trace_remove("write", trace_id)
                    self.data_type_var.set(detected_type)
                    self.data_type_var.trace_add("write", self._on_data_type_change)
        else:
            self.current_data_type = data_type

        # Update operators
        self._update_operators()

    def _update_operators(self):
        """Update operator dropdown based on current data type."""
        if self.current_data_type == "numeric":
            operators = list(self.NUMERIC_OPERATORS.keys())
        elif self.current_data_type == "date":
            operators = list(self.DATE_OPERATORS.keys())
        else:
            operators = list(self.TEXT_OPERATORS.keys())

        # Update dropdown menu
        menu = self.operator_dropdown["menu"]
        menu.delete(0, "end")

        for operator in operators:
            menu.add_command(
                label=operator, command=tk._setit(self.operator_var, operator)
            )

        # Set default operator
        if operators and (
            not self.operator_var.get() or self.operator_var.get() not in operators
        ):
            self.operator_var.set(operators[0])

    def _on_operator_change(self, *args):
        """Handle operator selection change."""
        operator = self.operator_var.get()
        if not operator:
            return

        # Clean up any traces before destroying widgets
        for widget in self.value_frame.winfo_children():
            # Check if widget has trace info stored
            if hasattr(widget, "trace_id") and hasattr(widget, "trace_var"):
                try:
                    widget.trace_var.trace_remove("write", widget.trace_id)
                except:
                    pass  # Trace might already be removed
            widget.destroy()

        # Create appropriate input based on operator
        if operator in ["is null", "not null"]:
            # No input needed
            tk.Label(
                self.value_frame,
                text="(no value needed)",
                font=self.fonts["small"],
                fg=self.theme_colours["text"],
                bg=self.theme_colours["background"],
            ).pack(side=tk.LEFT)

        elif operator == "between":
            # Two inputs for range — date or numeric
            if self.current_data_type == "date":
                self._create_date_between_input()
            else:
                self._create_between_input()

        elif operator in ["in", "not in"]:
            # Multi-select or text input
            self._create_multi_select_input()

        elif self.current_data_type == "date" and operator not in ["in", "not in"]:
            # Date-specific single input (calendar picker or text entry)
            self._create_date_input()

        else:
            # Single input (numeric or text)
            self._create_single_input()

    def _create_single_input(self):
        """Create single value input with autocomplete dropdown."""
        column = self.column_var.get()

        if self.current_data_type == "numeric":
            # Numeric entry
            entry = self._create_numeric_entry(self.value_frame, self.value_var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        else:
            # Text with autocomplete dropdown
            unique_vals = self.unique_values_cache.get(column, [])
            print(
                f"DEBUG _create_single_input: column='{column}', found {len(unique_vals)} unique values"
            )
            if unique_vals:
                print(f"  First few values: {unique_vals[:5]}")

            # Create autocomplete entry widget
            self._create_autocomplete_entry(
                self.value_frame, self.value_var, unique_vals
            )

    def _create_between_input(self):
        """Create inputs for between operator."""
        # Min value
        min_entry = self._create_numeric_entry(self.value_frame, self.value_var)
        min_entry.pack(side=tk.LEFT, padx=(0, 2))

        # Separator
        tk.Label(
            self.value_frame,
            text="and",
            font=self.fonts["small"],
            fg=self.theme_colours["text"],
            bg=self.theme_colours["background"],
        ).pack(side=tk.LEFT, padx=2)

        # Max value
        max_entry = self._create_numeric_entry(self.value_frame, self.value2_var)
        max_entry.pack(side=tk.LEFT, padx=(2, 0))

    def _create_date_input(self):
        """Create a single date input — calendar picker if available, text entry fallback."""
        if _HAS_DATE_ENTRY:
            try:
                date_widget = create_themed_date_entry(
                    self.value_frame,
                    textvariable=self.value_var,
                    theme_colors=self.theme_colours,
                    font=self.fonts["normal"],
                    width=12,
                    date_pattern="yyyy-mm-dd",
                )
                date_widget.pack(side=tk.LEFT, padx=(0, 2))
                return
            except Exception as e:
                print(f"DEBUG: DateEntry creation failed, using text fallback: {e}")

        # Fallback: plain text entry with placeholder hint
        entry = self._create_text_entry(self.value_frame, self.value_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Add hint label
        tk.Label(
            self.value_frame,
            text="yyyy-mm-dd",
            font=self.fonts["small"],
            fg=self.theme_colours.get("text_secondary", self.theme_colours["text"]),
            bg=self.theme_colours["background"],
        ).pack(side=tk.LEFT, padx=(4, 0))

    def _create_date_between_input(self):
        """Create two date inputs for the between operator."""
        if _HAS_DATE_ENTRY:
            try:
                date1 = create_themed_date_entry(
                    self.value_frame,
                    textvariable=self.value_var,
                    theme_colors=self.theme_colours,
                    font=self.fonts["normal"],
                    width=12,
                    date_pattern="yyyy-mm-dd",
                )
                date1.pack(side=tk.LEFT, padx=(0, 2))

                tk.Label(
                    self.value_frame,
                    text="and",
                    font=self.fonts["small"],
                    fg=self.theme_colours["text"],
                    bg=self.theme_colours["background"],
                ).pack(side=tk.LEFT, padx=2)

                date2 = create_themed_date_entry(
                    self.value_frame,
                    textvariable=self.value2_var,
                    theme_colors=self.theme_colours,
                    font=self.fonts["normal"],
                    width=12,
                    date_pattern="yyyy-mm-dd",
                )
                date2.pack(side=tk.LEFT, padx=(2, 0))
                return
            except Exception as e:
                print(f"DEBUG: DateEntry between creation failed, using text fallback: {e}")

        # Fallback: two text entries
        entry1 = self._create_text_entry(self.value_frame, self.value_var)
        entry1.pack(side=tk.LEFT, padx=(0, 2))

        tk.Label(
            self.value_frame,
            text="and",
            font=self.fonts["small"],
            fg=self.theme_colours["text"],
            bg=self.theme_colours["background"],
        ).pack(side=tk.LEFT, padx=2)

        entry2 = self._create_text_entry(self.value_frame, self.value2_var)
        entry2.pack(side=tk.LEFT, padx=(2, 0))

        # Hint
        tk.Label(
            self.value_frame,
            text="yyyy-mm-dd",
            font=self.fonts["small"],
            fg=self.theme_colours.get("text_secondary", self.theme_colours["text"]),
            bg=self.theme_colours["background"],
        ).pack(side=tk.LEFT, padx=(4, 0))

    def _create_multi_select_input(self):
        """Create multi-select input for IN/NOT IN operators."""
        column = self.column_var.get()
        unique_vals = self.unique_values_cache.get(column, [])

        # Main frame
        input_frame = tk.Frame(self.value_frame, bg=self.theme_colours["background"])
        input_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Text entry for comma-separated values
        entry = self._create_text_entry(input_frame, self.value_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        if unique_vals and len(unique_vals) < 100:
            # Button to show multi-select dialog
            select_btn = ModernButton(
                input_frame,
                text="Select...",
                command=lambda: self._show_multi_select_dialog(unique_vals),
                color=self.theme_colours["accent_blue"],
                theme_colors=self.theme,
            )
            select_btn.pack(side=tk.LEFT, padx=(5, 0))

    def _create_numeric_entry(self, parent, variable):
        """Create a numeric entry field with validation."""
        entry_frame = tk.Frame(
            parent,
            bg=self.theme_colours["field_bg"],
            highlightbackground=self.theme_colours["field_border"],
            highlightthickness=1,
        )

        entry = tk.Entry(
            entry_frame,
            textvariable=variable,
            bg=self.theme_colours["field_bg"],
            fg=self.theme_colours["text"],
            insertbackground=self.theme_colours["text"],
            font=self.fonts["normal"],
            bd=0,
            width=8,  # Reduced width
        )
        entry.pack(padx=3, pady=2)

        # Validation for numeric input
        def validate(*args):
            value = variable.get()
            if not value or value in ["-", ".", "-."]:
                return

            try:
                float(value)
            except ValueError:
                # Clean non-numeric characters
                cleaned = ""
                has_decimal = False
                for i, char in enumerate(value):
                    if char == "-" and i == 0:
                        cleaned += char
                    elif char == "." and not has_decimal:
                        cleaned += char
                        has_decimal = True
                    elif char.isdigit():
                        cleaned += char
                variable.set(cleaned)

        variable.trace_add("write", validate)
        return entry_frame

    def _create_text_entry(self, parent, variable):
        """Create a text entry field."""
        entry_frame = tk.Frame(
            parent,
            bg=self.theme_colours["field_bg"],
            highlightbackground=self.theme_colours["field_border"],
            highlightthickness=1,
        )

        entry = tk.Entry(
            entry_frame,
            textvariable=variable,
            bg=self.theme_colours["field_bg"],
            fg=self.theme_colours["text"],
            insertbackground=self.theme_colours["text"],
            font=self.fonts["normal"],
            bd=0,
            width=12,  # Reduced width
        )
        entry.pack(padx=3, pady=2)

        return entry_frame

    def _create_autocomplete_entry(self, parent, variable, suggestions):
        """Create an entry with autocomplete dropdown functionality."""
        # Main container
        container = tk.Frame(parent, bg=self.theme_colours["background"])
        container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Entry frame with border
        entry_frame = tk.Frame(
            container,
            bg=self.theme_colours["field_bg"],
            highlightbackground=self.theme_colours["field_border"],
            highlightthickness=1,
        )
        entry_frame.pack(fill=tk.X)

        # Entry widget
        entry = tk.Entry(
            entry_frame,
            textvariable=variable,
            bg=self.theme_colours["field_bg"],
            fg=self.theme_colours["text"],
            insertbackground=self.theme_colours["text"],
            font=self.fonts["normal"],
            bd=0,
            width=20,
        )
        entry.pack(padx=3, pady=2)

        # Dropdown listbox (initially hidden)
        dropdown_frame = None
        listbox = None

        def show_dropdown():
            nonlocal dropdown_frame, listbox

            # Don't show if no suggestions
            if not suggestions:
                print(f"DEBUG: No suggestions available for dropdown")
                return

            # Create dropdown if it doesn't exist
            if dropdown_frame is None:
                # Create dropdown as a toplevel overlay
                dropdown_frame = tk.Frame(
                    container,
                    bg=self.theme_colours["field_border"],
                    highlightbackground=self.theme_colours["field_border"],
                    highlightthickness=1,
                )

                # Create scrollbar and listbox
                scrollbar = tk.Scrollbar(dropdown_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

                listbox = tk.Listbox(
                    dropdown_frame,
                    bg=self.theme_colours["field_bg"],
                    fg=self.theme_colours["text"],
                    selectbackground=self.theme_colours["accent_blue"],
                    selectforeground="#ffffff",
                    font=self.fonts["small"],
                    height=min(8, len(suggestions)),
                    yscrollcommand=scrollbar.set,
                    bd=0,
                    highlightthickness=0,
                )
                listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.config(command=listbox.yview)

                # Populate with filtered suggestions
                update_dropdown()

                # Bind selection
                def on_select(event):
                    if listbox.curselection():
                        idx = listbox.curselection()[0]
                        value = listbox.get(idx)
                        variable.set(value)
                        hide_dropdown()

                listbox.bind("<<ListboxSelect>>", on_select)
                listbox.bind("<Return>", on_select)
                listbox.bind("<Double-Button-1>", on_select)

                # Add mouse wheel scrolling
                def on_mousewheel(event):
                    listbox.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    return "break"

                listbox.bind("<MouseWheel>", on_mousewheel)  # Windows
                listbox.bind(
                    "<Button-4>", lambda e: listbox.yview_scroll(-1, "units")
                )  # Linux up
                listbox.bind(
                    "<Button-5>", lambda e: listbox.yview_scroll(1, "units")
                )  # Linux down

            # Show the dropdown using place() to overlay instead of pushing widgets
            # Calculate position below the entry frame
            container.update_idletasks()  # Ensure geometry is calculated
            entry_height = entry_frame.winfo_height()
            
            dropdown_frame.place(
                x=0, 
                y=entry_height + 2,  # Position below entry with small gap
                relwidth=1.0,  # Match container width
                height=200  # Fixed height for dropdown
            )
            
            # Raise to top to ensure visibility
            dropdown_frame.lift()

        def hide_dropdown():
            nonlocal dropdown_frame
            if dropdown_frame:
                try:
                    dropdown_frame.winfo_exists()
                    dropdown_frame.place_forget()  # Use place_forget instead of pack_forget
                except tk.TclError:
                    # Widget has been destroyed
                    dropdown_frame = None

        def update_dropdown(*args):
            if not listbox:
                return

            # Check if listbox still exists (not destroyed)
            try:
                listbox.winfo_exists()
            except tk.TclError:
                # Widget has been destroyed, remove this trace
                try:
                    variable.trace_remove("write", trace_id)
                except:
                    pass
                return

            # Get current text
            current = variable.get().lower()

            # Clear listbox
            listbox.delete(0, tk.END)

            # Add filtered suggestions
            if current:
                # Filter suggestions
                filtered = [s for s in suggestions if current in str(s).lower()]
            else:
                # Show all suggestions
                filtered = suggestions[:50]  # Limit to first 50

            for item in filtered[:50]:  # Limit display
                listbox.insert(tk.END, str(item))

            # Adjust height
            listbox.config(height=min(8, len(filtered)))

        # Bind events
        def on_entry_click(event):
            show_dropdown()

        def on_entry_key(event):
            if dropdown_frame and dropdown_frame.winfo_ismapped():
                update_dropdown()
            else:
                show_dropdown()

        def on_focus_out(event):
            """Hide dropdown when focus moves away from entry."""
            # Check where focus went
            def check_and_hide():
                focus_widget = entry.focus_get()
                # Only hide if focus didn't go to the listbox
                if focus_widget != listbox and dropdown_frame:
                    try:
                        # Check if click was in dropdown area
                        if dropdown_frame.winfo_containing(entry.winfo_pointerx(), entry.winfo_pointery()) != dropdown_frame:
                            hide_dropdown()
                    except:
                        hide_dropdown()
            
            # Small delay to allow listbox click to register
            container.after(100, check_and_hide)

        entry.bind("<Button-1>", on_entry_click)
        entry.bind("<KeyRelease>", on_entry_key)
        entry.bind("<FocusOut>", on_focus_out)

        # Also update dropdown when variable changes
        trace_id = variable.trace_add("write", update_dropdown)

        # Store trace info for cleanup
        container.trace_id = trace_id
        container.trace_var = variable

        return container

    def _cache_unique_values(self, column):
        """Cache unique values for a column, handling nulls."""
        # Check if already cached and still valid
        if column in self.unique_values_cache:
            # Only re-cache if register data has changed
            if not hasattr(self, "_last_register_shape") or (
                self.register_data is not None
                and self._last_register_shape != self.register_data.shape
            ):
                # Data has changed, clear this column's cache
                del self.unique_values_cache[column]
            else:
                # Cache is still valid
                return

        if self.register_data is not None and not self.register_data.empty:
            self._last_register_shape = self.register_data.shape
        
        # Try callback first if available (allows parent to provide values from any source)
        if self.get_column_schema_callback:
            try:
                callback_values = self.get_column_schema_callback(column)
                if callback_values:
                    self.unique_values_cache[column] = callback_values[:200]
                    print(f"DEBUG: Got {len(callback_values)} values for '{column}' from callback")
                    return
            except Exception as e:
                print(f"DEBUG: Callback failed for '{column}': {e}")
        
        try:
            print(f"DEBUG _cache_unique_values: Checking column '{column}'")
            print(
                f"  Available columns in register_data: {list(self.register_data.columns)[:10]}"
            )
            
            # Strip source suffix for column lookup
            col_base = column.split(" (")[0].strip()
            
            # Try multiple column name variations
            col_variations = [column, col_base, col_base.lower(), col_base.upper()]
            found_col = None
            for var in col_variations:
                if var in self.register_data.columns:
                    found_col = var
                    break

            if found_col:
                # Get unique values - use callback if available (handles both CSV and register columns)
                if self.get_unique_values_callback:
                    unique_vals = self.get_unique_values_callback(column)
                else:
                    # Fallback to register_data (old behavior)
                    unique_vals = self.register_data[column].dropna().unique()

                # Convert to strings and sort, excluding None and nan strings
                str_vals = []
                for v in unique_vals:
                    # Skip None and NaN values
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        continue
                    str_v = str(v)
                    # Skip string representations of None/nan
                    if str_v.lower() in ["none", "nan", "null", ""]:
                        continue
                    str_vals.append(str_v)
                # Sort based on data type
                if self.current_data_type == "numeric":
                    try:
                        str_vals.sort(key=lambda x: float(x))
                    except Exception:
                        str_vals.sort()
                elif self.current_data_type == "date":
                    try:
                        def _date_sort_key(x):
                            dt = pd.to_datetime(x, errors='coerce')
                            # NaT sorts to the end
                            return (0, dt) if pd.notna(dt) else (1, x)
                        str_vals.sort(key=_date_sort_key)
                    except Exception:
                        str_vals.sort()
                else:
                    str_vals.sort()

                self.unique_values_cache[column] = str_vals[:200]  # Limit to 200

                print(
                    f"DEBUG: Cached {len(str_vals)} unique values for column '{column}'"
                )
        except Exception as e:
            print(f"ERROR caching values for column '{column}': {e}")
            self.unique_values_cache[column] = []

    def _show_dropdown_menu(self, entry_widget, values):
        """Show dropdown menu for single selection."""
        menu = tk.Menu(
            self.frame,
            tearoff=0,
            bg=self.theme_colours["field_bg"],
            fg=self.theme_colours["text"],
            font=self.fonts["small"],
        )

        for val in values[:50]:  # Limit display
            menu.add_command(label=val, command=lambda v=val: self.value_var.set(v))

        # Position menu below entry
        x = entry_widget.winfo_rootx()
        y = entry_widget.winfo_rooty() + entry_widget.winfo_height()
        menu.post(x, y)

    def _show_multi_select_dialog(self, values):
        """Show dialog for multi-selection."""
        current = self.value_var.get()
        current_list = [v.strip() for v in current.split(",")] if current else []

        # Create simple multi-select dialog
        dialog = tk.Toplevel(self.frame)
        dialog.title("Select Values")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()

        # Configure dialog appearance
        dialog.configure(bg=self.theme_colours["background"])
        dialog.geometry("400x500")

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 500) // 2
        dialog.geometry(f"+{x}+{y}")

        # Main frame
        main_frame = tk.Frame(dialog, bg=self.theme_colours["background"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Search entry
        search_frame = tk.Frame(main_frame, bg=self.theme_colours["background"])
        search_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            search_frame,
            text="Search:",
            bg=self.theme_colours["background"],
            fg=self.theme_colours["text"],
            font=self.fonts["normal"],
        ).pack(side=tk.LEFT, padx=(0, 5))

        search_var = tk.StringVar()
        search_entry = self._create_text_entry(search_frame, search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Scrollable list frame
        list_frame = tk.Frame(main_frame, bg=self.theme_colours["background"])
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for scrolling
        canvas = tk.Canvas(
            list_frame, bg=self.theme_colours["field_bg"], highlightthickness=0
        )
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame inside canvas
        inner_frame = tk.Frame(canvas, bg=self.theme_colours["field_bg"])
        canvas_window = canvas.create_window(0, 0, anchor="nw", window=inner_frame)

        # Track checkboxes
        check_vars = {}
        check_widgets = {}

        def update_list(*args):
            """Update displayed list based on search."""
            search_text = search_var.get().lower()

            # Clear existing widgets
            for widget in inner_frame.winfo_children():
                widget.destroy()
            check_widgets.clear()

            # Add filtered items
            row = 0
            for val in values[:100]:  # Limit to 100
                if search_text and search_text not in str(val).lower():
                    continue

                if val not in check_vars:
                    check_vars[val] = tk.BooleanVar(value=(str(val) in current_list))

                cb = tk.Checkbutton(
                    inner_frame,
                    text=str(val),
                    variable=check_vars[val],
                    bg=self.theme_colours["field_bg"],
                    fg=self.theme_colours["text"],
                    activebackground=self.theme_colours["field_bg"],
                    activeforeground=self.theme_colours["accent_blue"],
                    selectcolor=self.theme_colours["field_bg"],
                    font=self.fonts["small"],
                )
                cb.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                check_widgets[val] = cb
                row += 1

            # Update scroll region
            inner_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        # Bind search
        search_var.trace_add("write", update_list)

        # Initial population
        update_list()

        # Configure canvas width
        def configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", configure_canvas)

        # Button frame
        button_frame = tk.Frame(main_frame, bg=self.theme_colours["background"])
        button_frame.pack(fill=tk.X, pady=(10, 0))

        # Select/Deselect all buttons
        select_frame = tk.Frame(button_frame, bg=self.theme_colours["background"])
        select_frame.pack(side=tk.LEFT)

        def select_all():
            for var in check_vars.values():
                var.set(True)

        def select_none():
            for var in check_vars.values():
                var.set(False)

        ModernButton(
            select_frame,
            text="All",
            command=select_all,
            color=self.theme_colours["secondary_bg"],
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=2)

        ModernButton(
            select_frame,
            text="None",
            command=select_none,
            color=self.theme_colours["secondary_bg"],
            theme_colors=self.theme,
        ).pack(side=tk.LEFT, padx=2)

        # OK/Cancel buttons
        result = {"values": []}

        def on_ok():
            result["values"] = [
                str(val) for val, var in check_vars.items() if var.get()
            ]
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ModernButton(
            button_frame,
            text="Cancel",
            command=on_cancel,
            color=self.theme_colours["secondary_bg"],
            theme_colors=self.theme,
        ).pack(side=tk.RIGHT, padx=(5, 0))

        ModernButton(
            button_frame,
            text="OK",
            command=on_ok,
            color=self.theme_colours["accent_green"],
            theme_colors=self.theme,
        ).pack(side=tk.RIGHT)

        # Wait for dialog
        dialog.wait_window()

        # Update value if changed
        if result["values"]:
            self.value_var.set(", ".join(result["values"]))

    def get_filter_config(self):
        """Get the current filter configuration."""
        operator = self.operator_var.get()
        return {
            "column": self.column_var.get(),
            "data_type": self.current_data_type,
            "operator": operator,
            "value": self.value_var.get(),
            "value2": (
                self.value2_var.get() if operator == "between" else None
            ),
        }

    def apply_filter(self, row_data):
        """Apply this filter to a data row."""
        config = self.get_filter_config()
        column = config["column"]
        operator = config["operator"]
        value = config["value"]

        if column not in row_data:
            return False

        row_value = row_data[column]

        # Handle null operators
        if operator == "is null":
            return pd.isna(row_value)
        elif operator == "not null":
            return pd.notna(row_value)

        # Skip null values for other operators
        # Handle both None and NaN
        if row_value is None or pd.isna(row_value):
            return False

        # Apply operator based on type
        if self.current_data_type == "numeric":
            return self._apply_numeric_filter(
                row_value, operator, value, config.get("value2")
            )
        elif self.current_data_type == "date":
            return self._apply_date_filter(
                row_value, operator, value, config.get("value2")
            )
        else:
            return self._apply_text_filter(str(row_value), operator, value)

    def _apply_numeric_filter(self, row_value, operator, value, value2=None):
        """Apply numeric filter."""
        try:
            row_val = float(row_value)
            filter_val = float(value) if value else 0

            if operator == "=":
                return row_val == filter_val
            elif operator == "≠":
                return row_val != filter_val
            elif operator == "<":
                return row_val < filter_val
            elif operator == "≤":
                return row_val <= filter_val
            elif operator == ">":
                return row_val > filter_val
            elif operator == "≥":
                return row_val >= filter_val
            elif operator == "between" and value2:
                return float(value) <= row_val <= float(value2)
        except (ValueError, TypeError):
            return False

        return False

    def _apply_text_filter(self, row_value, operator, value):
        """Apply text filter."""
        row_val = str(row_value).lower()
        filter_val = str(value).lower() if value else ""

        if operator == "equals":
            return row_val == filter_val
        elif operator == "not equals":
            return row_val != filter_val
        elif operator == "contains":
            return filter_val in row_val
        elif operator == "not contains":
            return filter_val not in row_val
        elif operator == "starts with":
            return row_val.startswith(filter_val)
        elif operator == "ends with":
            return row_val.endswith(filter_val)
        elif operator == "in":
            values = [v.strip().lower() for v in value.split(",")]
            return row_val in values
        elif operator == "not in":
            values = [v.strip().lower() for v in value.split(",")]
            return row_val not in values
        elif operator == "like":
            # Simple pattern matching (% for wildcard)
            import re

            pattern = filter_val.replace("%", ".*")
            return bool(re.match(pattern, row_val))

        return False

    def _parse_date_value(self, value) -> 'datetime | None':
        """
        Parse a value into a datetime object.
        
        Handles: datetime objects, pd.Timestamp, date objects, and string formats.
        Returns None if parsing fails.
        """
        if value is None:
            return None

        # Already a datetime-like object
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, date_type):
            return datetime(value.year, value.month, value.day)

        # String parsing
        s = str(value).strip()
        if not s or s.lower() in ('', 'none', 'nan', 'null', 'nat'):
            return None

        # Try pandas parser first (handles most formats)
        try:
            result = pd.to_datetime(s, errors='raise')
            if pd.notna(result):
                return result.to_pydatetime()
        except Exception:
            pass

        # Try explicit formats as fallback
        for fmt in self.DATE_PARSE_FORMATS:
            try:
                return datetime.strptime(s, fmt)
            except (ValueError, TypeError):
                continue

        return None

    def _apply_date_filter(self, row_value, operator, value, value2=None):
        """Apply a date filter comparison."""
        row_date = self._parse_date_value(row_value)
        if row_date is None:
            return False

        filter_date = self._parse_date_value(value)
        if filter_date is None:
            return False

        # Compare date-only (strip time component for day-level operators)
        row_d = row_date.date()
        filter_d = filter_date.date()

        if operator == "on":
            return row_d == filter_d
        elif operator == "not on":
            return row_d != filter_d
        elif operator == "before":
            return row_d < filter_d
        elif operator == "on or before":
            return row_d <= filter_d
        elif operator == "after":
            return row_d > filter_d
        elif operator == "on or after":
            return row_d >= filter_d
        elif operator == "between":
            filter_date2 = self._parse_date_value(value2)
            if filter_date2 is None:
                return False
            filter_d2 = filter_date2.date()
            return row_d >= filter_d and row_d <= filter_d2

        return False

    def debug_theme_info(self):
        """Debug method to print current theme information."""
        print("\n=== THEME DEBUG INFO ===")
        print(f"Theme colors: {self.theme}")

        # Check styles on dropdowns
        for dropdown_name, dropdown in [
            ("column", self.column_dropdown),
            ("type", self.type_dropdown),
            ("operator", self.operator_dropdown),
        ]:
            print(f"\n{dropdown_name} dropdown:")
            print(f"  bg: {dropdown.cget('bg')}")
            print(f"  fg: {dropdown.cget('fg')}")
            print(f"  activebackground: {dropdown.cget('activebackground')}")
            print(f"  activeforeground: {dropdown.cget('activeforeground')}")

        print("======================\n")

    def update_columns(self, new_columns_info, new_register_data):
        """Update the available columns after CSV data is loaded."""
        self.columns_info = new_columns_info if new_columns_info else {}
        self.register_data = new_register_data

        # Clear the cache since we have new data
        self.unique_values_cache.clear()

        # Get column list
        if not self.columns_info and self.register_data is not None:
            columns = list(self.register_data.columns)
        else:
            columns = list(self.columns_info.keys())

        # Save current selection
        current_selection = self.column_var.get()

        # Update the dropdown menu
        # Check if using searchable dropdown or standard dropdown
        if hasattr(self.column_dropdown, 'set_items'):
            # Searchable dropdown - has set_items method
            self.column_dropdown.set_items(columns)
        else:
            # Standard dropdown - update menu directly
            menu = self.column_dropdown["menu"]
            menu.delete(0, "end")
            
            for column in columns:
                menu.add_command(label=column, command=tk._setit(self.column_var, column))

        # Keep current selection if it's still valid, otherwise set first column
        if current_selection in columns:
            self.column_var.set(current_selection)
            # Re-cache values for current column
            self._cache_unique_values(current_selection)
        elif columns:
            self.column_var.set(columns[0])
            self._on_column_change()  # Trigger update for new column

        print(f"Updated filter columns to: {columns[:10]}...")
        print(f"Register data has {len(self.register_data)} rows")

        # Trigger operator change to refresh value input
        self._on_operator_change()

    def destroy(self):
        """Destroy this filter row."""
        self.frame.destroy()
