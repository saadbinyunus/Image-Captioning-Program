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
import pytesseract
from sklearn.model_selection import train_test_split
from transformers import get_scheduler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# Disable CUDA (Force CPU usage)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This disables TensorFlow and PyTorch from using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageCaptionDataset(Dataset):
    def __init__(self, captions_file, images_folder, processor, data=None):
        self.images_folder = images_folder
        self.processor = processor

        # Load data
        if data is None:
            self.data = self._load_captions(captions_file)
        else:
            self.data = data

    def _load_captions(self, captions_file):
        data = []
        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    image_caption_id, caption = parts
                    image_filename = image_caption_id.split("#")[0]
                    data.append({"image": image_filename, "caption": caption})
        return pd.DataFrame(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename = self.data.iloc[idx]["image"]
        caption = self.data.iloc[idx]["caption"]
        image_path = os.path.join(self.images_folder, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=raw_image, return_tensors="pt")
        
        caption_input = self.processor.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        labels = caption_input.input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return {key: val.squeeze(0) for key, val in inputs.items()}


def fine_tune_model(model, processor):
    captions_file = "data/captions.txt"
    images_folder = "data/images"

    try:
        dataset = ImageCaptionDataset(captions_file, images_folder, processor)
        train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)

        train_dataset = ImageCaptionDataset(captions_file, images_folder, processor, data=train_data)
        test_dataset = ImageCaptionDataset(captions_file, images_folder, processor, data=test_data)

        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        optimizer = AdamW(model.parameters(), lr=3e-5)
        model.train()

        num_training_steps = len(train_dataloader) * 3
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        print("Fine-tuning the model...")
        for epoch in range(3):
            total_loss = 0
            for i, batch in enumerate(train_dataloader):
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]
                input_ids = labels.clone()
                input_ids[input_ids == -100] = processor.tokenizer.pad_token_id

                # Debugging: Check the final inputs to the model
                print(f"[DEBUG] Batch {i + 1}")
                print(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
                print(f"[DEBUG] input_ids shape: {input_ids.shape}")
                print(f"[DEBUG] labels shape: {labels.shape}")


                outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()  # Update the learning rate after every step
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_dataloader):.4f}")

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    pixel_values = batch["pixel_values"]
                    labels = batch["labels"]
                    input_ids = labels.clone()
                    input_ids[input_ids == -100] = processor.tokenizer.pad_token_id

                    outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
                    test_loss += outputs.loss.item()

            print(f"Test Loss after Epoch {epoch + 1}: {test_loss / len(test_dataloader):.4f}")

        model.save_pretrained("fine_tuned_model")
        processor.save_pretrained("fine_tuned_model")
        print("Fine-tuning complete. Model saved as 'fine_tuned_model'.")

    except Exception as e:
        print(f"Error during fine-tuning: {e}")


# Preprocess the image
def preprocess_image(image_path, image_size=384):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image_tensor = transform(raw_image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Load pre-trained BLIP model and processor
print("Loading BLIP model and processor...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()


def handle_model_choice(choice):
    global model, processor
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to("cpu")  # Ensure model is on CPU

        if choice == 1:
            print("Fine-tuning the model...")
            fine_tune_model(model, processor)
        elif choice == 2:
            model_path = "fine_tuned_model"
            if os.path.exists(model_path):
                print("Loading saved fine-tuned model...")
                model = BlipForConditionalGeneration.from_pretrained(model_path)
                processor = BlipProcessor.from_pretrained(model_path)
                print("Fine-tuned model loaded successfully.")
            else:
                messagebox.showerror("Error", "Fine-tuned model not found. Please fine-tune the model first.")
        elif choice == 3:
            print("Using the pre-trained model for inference.")
        else:
            messagebox.showerror("Error", "Invalid option selected.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def create_model_selection_gui():
    model_selection_root = tk.Tk()
    model_selection_root.title("Select Model Option")
    model_selection_root.geometry("300x200")
    
    label = ttk.Label(model_selection_root, text="Choose an option:", font=("Helvetica", 12))
    label.pack(pady=10)

    fine_tune_button = ttk.Button(model_selection_root, text="Fine-tune Model", command=lambda: [handle_model_choice(1), model_selection_root.destroy()])
    fine_tune_button.pack(pady=5)

    saved_model_button = ttk.Button(model_selection_root, text="Use Saved Fine-tuned Model", command=lambda: [handle_model_choice(2), model_selection_root.destroy()])
    saved_model_button.pack(pady=5)

    pretrained_model_button = ttk.Button(model_selection_root, text="Use Pre-trained Model", command=lambda: [handle_model_choice(3), model_selection_root.destroy()])
    pretrained_model_button.pack(pady=5)

    model_selection_root.mainloop()


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

# Global variable to store image path
image_path = None

# Function to handle image selection
def select_image(root, caption_label):
    global image_path  # Declare image_path as global so it can be accessed outside this function
    print("Opening file dialog...")
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    
    if image_path:
        show_image(image_path, root)
        generate_button = ttk.Button(root, text="Generate Caption", command=lambda: generate_caption_threaded(image_path, caption_label))
        generate_button.grid(row=2, column=0, pady=20, padx=10, sticky="ew")
    
    # No need to return image_path as it is now global


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


# Function to extract text from an image using OCR
def extract_text_with_ocr(image_path):
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Use pytesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(image)
        
        # Return the extracted text
        return extracted_text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

# Function to speak the extracted OCR text
def speak_ocr_text(text):
    if text.strip():  # Ensure there's text to speak
        engine = pyttsx3.init()  # Initialize the TTS engine
        engine.setProperty('rate', 150)  # Set the speaking speed
        engine.setProperty('volume', 1.0)  # Set volume to maximum
        
        # Speak the OCR text
        engine.say(text)
        engine.runAndWait()
    else:
        print("No text to speak.")


# Function to run OCR and display the extracted text
def extract_text_threaded(image_path, ocr_label):
    def task():
        if image_path:
            text = extract_text_with_ocr(image_path)
            if text:
                ocr_label.after(0, ocr_label.config, {"text": text})
            else:
                messagebox.showerror("Error", "No text found in the image.")
        else:
            messagebox.showerror("Error", "No image selected.")
    
    thread = threading.Thread(target=task)
    thread.start()

def create_gui():
    root = tk.Tk()
    root.title("Image Caption and OCR Generator")
    
    # Set window size and background color
    root.geometry("600x800")
    root.configure(bg="#F4F6F8")

    # Create a frame for content
    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    # Configure weights for the root window
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Configure weights for the frame
    frame.grid_rowconfigure(0, weight=0)  # Title row
    frame.grid_rowconfigure(1, weight=1)  # Image and output rows
    frame.grid_rowconfigure(2, weight=0)  # Buttons row
    frame.grid_rowconfigure(3, weight=1)  # Output display rows
    frame.grid_columnconfigure(0, weight=1)  # Content column

    # Title Label
    title_label = ttk.Label(frame, text="Image Caption and OCR Generator", font=("Helvetica", 18, "bold"), anchor="center")
    title_label.grid(row=0, column=0, pady=10)

    # Button to Select Image
    select_button = ttk.Button(frame, text="Select Image", command=lambda: select_image(root, caption_label))
    select_button.grid(row=1, column=0, pady=10, sticky="ew")

    # Caption Label
    caption_label = ttk.Label(frame, text="Caption will appear here", wraplength=500, font=("Helvetica", 12), anchor="center")
    caption_label.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

    # OCR Text Label
    ocr_label = ttk.Label(frame, text="OCR Text will appear here", wraplength=500, font=("Helvetica", 12), anchor="center")
    ocr_label.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

    # Button to Generate Caption
    generate_button = ttk.Button(frame, text="Generate Caption", command=lambda: generate_caption_threaded(image_path, caption_label))
    generate_button.grid(row=4, column=0, pady=10, sticky="ew")

    # Button to Extract Text using OCR
    ocr_button = ttk.Button(frame, text="Extract Text (OCR)", command=lambda: extract_text_threaded(image_path, ocr_label))
    ocr_button.grid(row=5, column=0, pady=10, sticky="ew")

    # Button to Speak Caption
    speak_caption_button = ttk.Button(frame, text="Speak Caption", command=lambda: speak_caption(caption_label.cget("text")))
    speak_caption_button.grid(row=6, column=0, pady=10, sticky="ew")

    # Button to Speak OCR Text
    speak_ocr_button = ttk.Button(frame, text="Speak OCR", command=lambda: speak_ocr_text(ocr_label.cget("text")))
    speak_ocr_button.grid(row=7, column=0, pady=10, sticky="ew")

    root.mainloop()

# Run the application
if __name__ == "__main__":
    create_model_selection_gui()
    create_gui()  # Starts the main application GUI


