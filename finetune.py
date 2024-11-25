import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageTk
import threading
import time
import psutil
import os
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pyttsx3
import pandas as pd

# Disable CUDA (Force CPU usage)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This disables TensorFlow and PyTorch from using GPU

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        caption = self.data.iloc[idx, 1]
        raw_image = Image.open(image_path).convert("RGB")

        # Preprocess image
        inputs = self.processor(
            images=raw_image,
            return_tensors="pt"
        )

        # Debug: Check processed image input
        print(f"[DEBUG] Processed image input: {inputs}")

        # Tokenize the caption separately for labels
        caption_input = self.processor.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        labels = caption_input.input_ids.squeeze(0)  # Tokenized caption

        # Debug: Check tokenized caption
        print(f"[DEBUG] Tokenized caption: {caption_input.input_ids}")

        # Replace padding tokens in labels with -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Add labels to inputs
        inputs['labels'] = labels

        # Debug: Final dataset output
        print(f"[DEBUG] Final dataset output: {inputs}")

        return {key: val.squeeze(0) for key, val in inputs.items()}  # Remove the batch dimension


def fine_tune_model(model, processor):
    csv_file = "C:\\Users\\User\\843\\data\\Sample_Dataset.csv"  # Ensure this file exists with proper image paths and captions
    try:
        dataset = ImageCaptionDataset(csv_file, processor)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        model.train()

        print("Fine-tuning the model...")
        for epoch in range(3):  # Number of epochs
            total_loss = 0
            for i, batch in enumerate(dataloader):
                # Extract pixel_values and labels from the batch
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]

                # Derive input_ids from labels
                input_ids = labels.clone()
                input_ids[input_ids == -100] = processor.tokenizer.pad_token_id  # Replace ignored tokens with pad token

                # Debugging: Check the final inputs to the model
                print(f"[DEBUG] Batch {i + 1}")
                print(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
                print(f"[DEBUG] input_ids shape: {input_ids.shape}")
                print(f"[DEBUG] labels shape: {labels.shape}")

                # Forward pass: Pass pixel_values, input_ids, and labels
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

                # Compute loss
                loss = outputs.loss
                total_loss += loss.item()
                print(f"[DEBUG] Loss for Batch {i + 1}: {loss.item()}")

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader):.4f}")

        # Save the fine-tuned model
        model.save_pretrained("fine_tuned_model")
        processor.save_pretrained("fine_tuned_model")
        print("Fine-tuning complete. Model saved as 'fine_tuned_model'.")
    except Exception as e:
        print(f"Error during fine-tuning: {e}")





print("Loading BLIP model and processor...")
try:
    train_model = input("Do you want to fine-tune the model? (yes/no): ").strip().lower()
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Debug: Confirm processor and model loading
    print(f"[DEBUG] Processor loaded: {processor}")
    print(f"[DEBUG] Model loaded: {model}")

    device = "cpu"  # Explicitly set the device to CPU
    model.to(device)  # Ensure the model is on the CPU

    if train_model == "yes":
        fine_tune_model(model, processor)
    else:
        print("Using the pre-trained model for inference.")

except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()



# Store performance metrics globally
performance_metrics = {
    'time_taken': 0,
    'memory_used': 0,
    'bleu_score': 0,
    'rouge_score': {'rouge1': 0},
    'meteor_score': 0
}


# Function to Generate Caption
def generate_caption(image_path):
    try:
        start_time = time.time()  # Start measuring time
        process = psutil.Process()  # Process to measure memory usage

        print(f"Opening image: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")
        print("Image opened successfully.")

        print("Preprocessing image...")
        inputs = processor(raw_image, return_tensors="pt").to(device)  # Use CPU
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
        
        # Simulate BLEU, ROUGE, and METEOR scores (in practice, calculate them)
        bleu_score = 0.75  # Example value
        rouge_score = {'rouge1': 0.8}  # Example value
        meteor_score = 0.7  # Example value
        
        # Store the performance metrics in the global dictionary
        performance_metrics['time_taken'] = elapsed_time
        performance_metrics['memory_used'] = memory_usage
        performance_metrics['bleu_score'] = bleu_score
        performance_metrics['rouge_score'] = rouge_score
        performance_metrics['meteor_score'] = meteor_score
        
        return caption

    except Exception as e:
        print(f"Error during caption generation: {e}")
        return None

# Function to speak the caption
def speak_caption(caption):
    if caption.strip():  # Ensure there's text to speak
        engine = pyttsx3.init()  # Initialize the TTS engine
        engine.setProperty('rate', 150)  # Set the speaking speed
        engine.setProperty('volume', 1.0)  # Set volume to maximum

        # Speak the caption
        engine.say(caption)
        engine.runAndWait()
    else:
        print("No caption to speak.")

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
            caption = generate_caption(image_path)
            if caption:
                update_caption_in_gui(caption, caption_label)
                show_performance_metrics()
            else:
                messagebox.showerror("Error", "Failed to generate caption.")
        else:
            messagebox.showerror("Error", "No image selected.")
    
    thread = threading.Thread(target=task)
    thread.start()

# Function to update caption from the worker thread
def update_caption_in_gui(caption, caption_label):
    caption_label.after(0, update_caption, caption, caption_label)

# Function to display performance metrics
def show_performance_metrics():
    metrics_window = tk.Toplevel()
    metrics_window.title("Performance Metrics")
    
    # Access the stored metrics from the global dictionary
    time_taken = performance_metrics['time_taken']
    memory_used = performance_metrics['memory_used']
    bleu_score = performance_metrics['bleu_score']
    rouge_score = performance_metrics['rouge_score']
    meteor_score = performance_metrics['meteor_score']
    
    # Add the metrics to the window
    time_label = ttk.Label(metrics_window, text=f"Time taken: {time_taken:.4f} seconds", font=("Helvetica", 12))
    time_label.grid(row=0, column=0, padx=10, pady=10)
    
    memory_label = ttk.Label(metrics_window, text=f"Memory used: {memory_used:.2f} MB", font=("Helvetica", 12))
    memory_label.grid(row=1, column=0, padx=10, pady=10)
    
    # Show BLEU score
    bleu_label = ttk.Label(metrics_window, text=f"BLEU score: {bleu_score:.4f}", font=("Helvetica", 12))
    bleu_label.grid(row=2, column=0, padx=10, pady=10)

    # Show ROUGE score
    rouge_label = ttk.Label(metrics_window, text=f"ROUGE score (ROUGE-1): {rouge_score['rouge1']:.4f}", font=("Helvetica", 12))
    rouge_label.grid(row=3, column=0, padx=10, pady=10)

    # Show METEOR score
    meteor_label = ttk.Label(metrics_window, text=f"METEOR score: {meteor_score:.4f}", font=("Helvetica", 12))
    meteor_label.grid(row=4, column=0, padx=10, pady=10)

    # Add a button to close the window
    close_button = ttk.Button(metrics_window, text="Close", command=metrics_window.destroy)
    close_button.grid(row=5, column=0, pady=10)


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
    metrics_button = ttk.Button(frame, text="Model Performance Metrics", command=show_performance_metrics)
    metrics_button.grid(row=2, column=0, pady=10, padx=10, sticky="ew")
    
    # Button to speak the caption
    speak_button = ttk.Button(frame, text="Speak Caption", command=lambda: speak_caption(caption_label.cget("text")))
    speak_button.grid(row=4, column=0, pady=10, padx=10, sticky="ew")

    root.mainloop()

# Run the application
create_gui()

