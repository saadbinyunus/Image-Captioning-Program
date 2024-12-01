import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageTk
import threading
import time
import psutil
import pytesseract
import os
import pytesseract

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Load pre-trained BLIP model and processor
print("Loading BLIP model and processor...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

# Global variable to store image path
image_path = None

# Function to generate a caption and measure performance
def generate_caption(image_path):
    try:
        start_time = time.time()  # Start measuring time
        process = psutil.Process()  # Process to measure memory usage

        print(f"Opening image: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")
        print("Image opened successfully.")

        print("Preprocessing image...")
        inputs = processor(raw_image, return_tensors="pt")
        print("Image preprocessed.")

        print("Generating caption...")
        out = model.generate(**inputs)
        print("Caption generated.")

        # Decode the output into text
        caption = processor.decode(out[0], skip_special_tokens=True)
        print("Caption decoded successfully.")
        
        # Measure time taken and memory usage
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
        
        return caption, elapsed_time, memory_usage

    except Exception as e:
        print(f"Error during caption generation: {e}")
        return None, 0, 0

# Function to handle image selection
def select_image(root, caption_label):
    global image_path  # Declare the global variable here
    print("Opening file dialog...")
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    
    if image_path:
        show_image(image_path, root)
        generate_button = ttk.Button(root, text="Generate Caption", command=lambda: generate_caption_threaded(image_path, caption_label))
        generate_button.grid(row=2, column=0, pady=20, padx=10, sticky="ew")
    
    return image_path

# Function to display the selected image
def show_image(image_path, root):
    img = Image.open(image_path)
    img.thumbnail((400, 400))  # Resize image
    img_display = ImageTk.PhotoImage(img)
    image_label = tk.Label(root, image=img_display, bd=2, relief="solid")
    image_label.image = img_display  # Keep a reference to the image
    image_label.grid(row=1, column=0, pady=20, sticky="nsew")
    
# Function to update caption in the GUI
def update_caption(caption, caption_label):
    caption_label.config(text=caption)

# Function to run caption generation in a separate thread
def generate_caption_threaded(image_path, caption_label):
    def task():
        if image_path:
            caption, time_taken, memory_used = generate_caption(image_path)
            if caption:
                update_caption(caption, caption_label)
                show_performance_metrics(time_taken, memory_used)
            else:
                messagebox.showerror("Error", "Failed to generate caption.")
        else:
            messagebox.showerror("Error", "No image selected.")
    
    thread = threading.Thread(target=task)
    thread.start()

# Function to display performance metrics in a new window (general metrics)
def show_performance_metrics(time_taken, memory_used):
    metrics_window = tk.Toplevel()
    metrics_window.title("Performance Metrics")
    
    # Add the metrics to the window
    time_label = ttk.Label(metrics_window, text=f"Time taken: {time_taken:.4f} seconds", font=("Helvetica", 12))
    time_label.grid(row=0, column=0, padx=10, pady=10)
    
    memory_label = ttk.Label(metrics_window, text=f"Memory used: {memory_used:.2f} MB", font=("Helvetica", 12))
    memory_label.grid(row=1, column=0, padx=10, pady=10)
    
    # Add a button to close the window
    close_button = ttk.Button(metrics_window, text="Close", command=metrics_window.destroy)
    close_button.grid(row=2, column=0, pady=10)

# Function to perform OCR (Optical Character Recognition) on the selected image
def perform_ocr(image_path, ocr_label):
    try:
        print(f"Performing OCR on image: {image_path}")
        img = Image.open(image_path)
        ocr_text = pytesseract.image_to_string(img)  # Extract text from image
        print("OCR completed.")
        if ocr_text.strip():  # If OCR text is not empty
            ocr_label.config(text=ocr_text)
        else:
            ocr_label.config(text="No text found in the image.")
    except Exception as e:
        print(f"Error during OCR: {e}")
        messagebox.showerror("OCR Error", "Failed to perform OCR.")

# Function to create the GUI
def create_gui():
    global image_path  # Declare the global variable here
    root = tk.Tk()
    root.title("Image Caption Generator & OCR Tool")
    
    # Set window size and background color
    root.geometry("600x800")
    root.configure(bg="#F4F6F8")

    # Create a frame for content
    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    # Grid row and column weight configuration for the root window
    root.grid_rowconfigure(0, weight=1)  # Allow content area to expand vertically
    root.grid_columnconfigure(0, weight=1)  # Allow content area to expand horizontally

    # Grid row and column weight configuration for the frame
    frame.grid_rowconfigure(0, weight=0)  # Title row
    frame.grid_rowconfigure(1, weight=1)  # Image display row (expands when window size changes)
    frame.grid_rowconfigure(2, weight=0)  # Button row
    frame.grid_rowconfigure(3, weight=1)  # Caption row
    frame.grid_rowconfigure(4, weight=1)  # OCR result row

    frame.grid_columnconfigure(0, weight=1)  # All widgets in the frame should expand horizontally

    # Label for the title (centered)
    title_label = ttk.Label(frame, text="Image Caption Generator & OCR Tool", font=("Helvetica", 18, "bold"), anchor="center")
    title_label.grid(row=0, column=0, columnspan=2, pady=10)

    # Button to select an image (centered)
    select_button = ttk.Button(frame, text="Select Image", command=lambda: select_image(root, caption_label))
    select_button.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

    # Label to display the caption (centered)
    caption_label = ttk.Label(frame, text="Caption will appear here", wraplength=500, font=("Helvetica", 12), anchor="center")
    caption_label.grid(row=3, column=0, pady=20, padx=10, sticky="ew")

    # Button to trigger OCR
    ocr_button = ttk.Button(frame, text="Perform OCR", command=lambda: perform_ocr(image_path, ocr_label))
    ocr_button.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

    # Label to display the OCR text result
    ocr_label = ttk.Label(frame, text="OCR result will appear here", wraplength=500, font=("Helvetica", 12), anchor="center")
    ocr_label.grid(row=4, column=0, pady=20, padx=10, sticky="ew")

    # Start the GUI loop
    root.mainloop()

# Start the GUI
create_gui()
