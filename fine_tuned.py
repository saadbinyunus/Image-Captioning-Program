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

# Disable CUDA (Force CPU usage)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This disables TensorFlow and PyTorch from using GPU

class ImageCaptionDataset(Dataset):
    def __init__(self, captions_file, images_folder, processor):
        self.images_folder = images_folder
        self.processor = processor
        self.data = self._load_captions(captions_file)

    def _load_captions(self, captions_file):
        data = []
        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    image_caption_id, caption = parts
                    image_filename = image_caption_id.split("#")[0]  # Extract the image filename
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
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        model.train()

        for epoch in range(3):  
            total_loss = 0
            for i, batch in enumerate(dataloader):
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]
                input_ids = labels.clone()
                input_ids[input_ids == -100] = processor.tokenizer.pad_token_id
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.save_pretrained("fine_tuned_model")
        processor.save_pretrained("fine_tuned_model")
        print("Fine-tuning complete. Model saved as 'fine_tuned_model'.")
    except Exception as e:
        print(f"Error during fine-tuning: {e}")


def load_model_choice():
    # Create a new window for model selection
    model_choice_window = tk.Toplevel()
    model_choice_window.title("Select Model Option")

    # Label for instructions
    instructions_label = ttk.Label(model_choice_window, text="Please select the model option:", font=("Helvetica", 12))
    instructions_label.pack(pady=10)

    def on_fine_tune():
        model_choice_window.destroy()
        fine_tune_model(model, processor)

    def on_use_finetuned():
        model_choice_window.destroy()
        processor = BlipProcessor.from_pretrained("fine_tuned_model")
        model = BlipForConditionalGeneration.from_pretrained("fine_tuned_model")
        print("Loaded fine-tuned model.")

    def on_use_pretrained():
        model_choice_window.destroy()
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Loaded pre-trained model.")

    # Buttons for model selection
    fine_tune_button = ttk.Button(model_choice_window, text="Fine-tune the model", command=on_fine_tune)
    fine_tune_button.pack(fill='x', pady=5)

    finetuned_button = ttk.Button(model_choice_window, text="Use previously fine-tuned model", command=on_use_finetuned)
    finetuned_button.pack(fill='x', pady=5)

    pretrained_button = ttk.Button(model_choice_window, text="Use pretrained model", command=on_use_pretrained)
    pretrained_button.pack(fill='x', pady=5)

    model_choice_window.mainloop()


# Main GUI and Functionality
def create_gui():
    root = tk.Tk()
    root.title("Image Caption and OCR Generator")
    root.geometry("600x800")
    root.configure(bg="#F4F6F8")

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    title_label = ttk.Label(frame, text="Image Caption and OCR Generator", font=("Helvetica", 18, "bold"), anchor="center")
    title_label.grid(row=0, column=0, pady=10)

    # Model selection at the start
    load_model_choice()

    root.mainloop()


# Run the application
create_gui()
