# AGENTS Instructions for GeoVue

This repository contains the GUI-based *GeoVue* project. When contributing to the codebase follow these rules:

## General Guidelines
- **Tkinter UI must run on the main thread.** The main/root window is created in `src/main.py`. Any window or dialog should be created from that thread.
- **Keep the codebase modular.** Functions or methods that could be reused should be placed in their own module within `src/` and documented with detailed docstrings.
- **File operations** must go through `core.FileManager`.
- **Theming** should always be handled by `gui.GUIManager`.
- **No CLI utilities** should be added. The application is GUI-only.
- **Use modern widgets** (e.g. `ModernButton`) for button creation.
- **Register interactions** go through the `Register Manager` (see `utils` modules).
- **Dialogs and translations** must use `gui.dialog_helper.DialogHelper`. Use `self.t()` or `DialogHelper.t()` for user-facing strings. Do not pass variables directly to the translation function; build strings first then translate.

## Code Snippet Format
When instructing how to insert or replace code, include at least one existing line before and after the change and use the following format:

```
[existing previous code line]
# ===================================================
# INSERT: explanation or new code
[new code]
# ===================================================
[existing next code line]
```

For replacements use:
```
[existing previous code line]
# ===================================================
# REMOVE: old code description
# REPLACE WITH: new code description
[new code]
# ===================================================
[existing next code line]
```

## Directory Layout
Refer to `core/file_manager.py` for the full local and shared directory structures. These must remain consistent with the documentation found in that file.

## Programmatic Checks
There are currently no automated tests. After modifying code run:
```
python -m compileall src
```
to ensure modules compile without syntax errors.
