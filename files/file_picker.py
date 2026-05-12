"""ai_files_v1.file_picker — native OS file picker (tkinter).

Direct port of v4.1 cell 11. No logic changes.
"""

from __future__ import annotations


def open_native_file_picker():
    """Opens the OS file-open dialog with multi-select enabled.
    Returns a tuple of selected paths, or () if cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        paths = filedialog.askopenfilenames(
            title="Select files to attach (Ctrl/Shift+click for multiple)",
            filetypes=[
                ("All supported", "*.docx *.pptx *.py *.txt *.md *.csv *.xlsx *.json *.html *.sql *.r *.js *.ts"),
                ("Word documents", "*.docx"),
                ("PowerPoint", "*.pptx"),
                ("Python", "*.py"),
                ("Text/Markdown", "*.txt *.md"),
                ("Data", "*.csv *.xlsx *.json"),
                ("All files", "*.*"),
            ]
        )
        root.destroy()
        return paths
    except Exception as e:
        print(f"⚠️  Native picker unavailable: {e}")
        return ()
