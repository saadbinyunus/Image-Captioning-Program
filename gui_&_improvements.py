import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageTk
import threading

# Load pre-trained BLIP model and processor
print("Loading BLIP model and processor...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

# Function to generate a caption
def generate_caption(image_path):
    try:
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
        return caption

    except Exception as e:
        print(f"Error during caption generation: {e}")
        return None

# Function to handle image selection
def select_image(root, caption_label):
    print("Opening file dialog...")
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    
    if image_path:
        show_image(image_path, root)
        generate_button = tk.Button(root, text="Generate Caption", command=lambda: generate_caption_threaded(image_path, caption_label))
        generate_button.pack(pady=10)
    
    return image_path

# Function to display the selected image
def show_image(image_path, root):
    img = Image.open(image_path)
    img.thumbnail((400, 400))  # Resize image
    img_display = ImageTk.PhotoImage(img)
    image_label = tk.Label(root, image=img_display)
    image_label.image = img_display  # Keep a reference to the image
    image_label.pack()

# Function to update caption in the GUI
def update_caption(caption, caption_label):
    caption_label.config(text=caption)

# Function to run caption generation in a separate thread
def generate_caption_threaded(image_path, caption_label):
    def task():
        if image_path:
            caption = generate_caption(image_path)
            if caption:
                update_caption(caption, caption_label)
            else:
                messagebox.showerror("Error", "Failed to generate caption.")
        else:
            messagebox.showerror("Error", "No image selected.")
    
    thread = threading.Thread(target=task)
    thread.start()

# Function to create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Image Caption Generator")

    # Button to select an image
    select_button = tk.Button(root, text="Select Image", command=lambda: select_image(root, caption_label))
    select_button.pack(pady=10)

    # Label to display the caption
    caption_label = tk.Label(root, text="Caption will appear here", wraplength=400)
    caption_label.pack(pady=20)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()
