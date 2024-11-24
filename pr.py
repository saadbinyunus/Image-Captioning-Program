import tkinter as tk
from tkinter import filedialog
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load pre-trained BLIP model and processor
print("Loading BLIP model and processor...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Model and processor loaded.")

# Function to generate a caption
def generate_caption(image_path):
    print("Opening image...")
    # Open the image
    try:
        raw_image = Image.open(image_path).convert("RGB")
        print("Image opened.")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    print("Preprocessing image...")
    # Preprocess the image and use the model to generate a caption
    try:
        inputs = processor(raw_image, return_tensors="pt")
        print("Image preprocessed.")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

    print("Generating caption...")
    try:
        out = model.generate(**inputs)
        print("Caption generated.")
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None

    # Decode the output into text
    try:
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error decoding caption: {e}")
        return None

# Function to open a file dialog and select an image
def select_image():
    print("Opening file dialog...")
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    print(f"Selected file: {file_path}")
    return file_path

# Test with a file selection dialog
print("Starting image caption generation...")
image_path = select_image()  # This will open the file picker
if image_path:
    print(f"Image selected: {image_path}")
    caption = generate_caption(image_path)
    if caption:
        print("Generated Caption:", caption)
    else:
        print("Failed to generate caption.")
else:
    print("No image selected.")
