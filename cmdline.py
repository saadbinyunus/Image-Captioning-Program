import tkinter as tk
from tkinter import filedialog

def test_file_dialog():
    root = tk.Tk()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    print(f"Selected file: {file_path}")
    root.destroy()

test_file_dialog()
