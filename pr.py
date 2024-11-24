import tkinter as tk
from tkinter import filedialog
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate a caption
def generate_caption(image_path):
    # Open the image
    raw_image = Image.open(image_path).convert("RGB")

    # Preprocess the image and use the model to generate a caption
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    
    # Decode the output into text
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to open a file dialog and select an image
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Test with a file selection dialog
image_path = select_image()  # This will open the file picker
if image_path:
    caption = generate_caption(image_path)
    print("Generated Caption:", caption)
else:
    print("No image selected.")
