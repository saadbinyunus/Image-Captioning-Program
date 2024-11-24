import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageTk
import threading
import time
import psutil
import os
import torch

# Load pre-trained BLIP model and processor
print("Loading BLIP model and processor...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

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

# Function to display ML model performance metrics in a new window
def show_model_performance_metrics():
    metrics_window = tk.Toplevel()
    metrics_window.title("Model Performance Metrics")
    
    # Get model file size using Hugging Face model attributes
    model_size = get_model_size()
    
    # Inference time (measure by dummy run if needed)
    start_time = time.time()
    
    # Create a dummy image for inference (a black image)
    dummy_image = Image.new('RGB', (224, 224), color='black')  # Create a dummy black image
    dummy_input = processor(dummy_image, return_tensors="pt")  # Pass the dummy image to processor
    _ = model.generate(**dummy_input)  # Perform dummy inference to measure time
    
    inference_time = time.time() - start_time
    
    # Display model size and inference time
    model_size_label = ttk.Label(metrics_window, text=f"Model Size: {model_size} MB", font=("Helvetica", 12))
    model_size_label.grid(row=0, column=0, padx=10, pady=10)
    
    inference_time_label = ttk.Label(metrics_window, text=f"Inference Time (Dummy): {inference_time:.4f} seconds", font=("Helvetica", 12))
    inference_time_label.grid(row=1, column=0, padx=10, pady=10)
    
    # Add a button to close the window
    close_button = ttk.Button(metrics_window, text="Close", command=metrics_window.destroy)
    close_button.grid(row=2, column=0, pady=10)

# Function to get the model size from Hugging Face directly
def get_model_size():
    try:
        # Number of parameters of the model
        model_size = sum(p.numel() for p in model.parameters()) / (1024 * 1024)  # in MB
        return round(model_size, 2)
    except Exception as e:
        print(f"Error getting model size: {e}")
        return 0

# Function to create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Image Caption Generator")
    
    # Set window size and background color
    root.geometry("600x700")
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

    frame.grid_columnconfigure(0, weight=1)  # All widgets in the frame should expand horizontally

    # Label for the title (centered)
    title_label = ttk.Label(frame, text="Image Caption Generator", font=("Helvetica", 18, "bold"), anchor="center")
    title_label.grid(row=0, column=0, columnspan=2, pady=10)

    # Button to select an image (centered)
    select_button = ttk.Button(frame, text="Select Image", command=lambda: select_image(root, caption_label))
    select_button.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

    # Label to display the caption (centered)
    caption_label = ttk.Label(frame, text="Caption will appear here", wraplength=500, font=("Helvetica", 12), anchor="center")
    caption_label.grid(row=3, column=0, pady=20, padx=10, sticky="ew")

    # Performance metrics button
    metrics_button = ttk.Button(frame, text="Model Performance Metrics", command=show_model_performance_metrics)
    metrics_button.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

    # Start the GUI loop
    root.mainloop()

# Start the GUI
create_gui()
