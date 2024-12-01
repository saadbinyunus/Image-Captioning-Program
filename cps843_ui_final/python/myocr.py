import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageTk
import threading
import time
import psutil
import pytesseract
import os
import pyttsx3
import nltk
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# Disable CUDA (Force CPU usage)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This disables TensorFlow and PyTorch from using GPU

# Temporarily took out Dataset arg
class ImageCaptionDataset(Dataset):
    def __init__(self, captions_file, images_folder, processor):
        """
        Args:
            captions_file (str): Path to the captions.txt file.
            images_folder (str): Path to the folder containing images.
            processor (BlipProcessor): Processor for tokenizing captions and preprocessing images.
        """
        self.images_folder = images_folder
        self.processor = processor

        # Load and parse captions
        self.data = self._load_captions(captions_file)

    def _load_captions(self, captions_file):
        """
        Parses the captions.txt file and returns a DataFrame with image filenames and captions.
        """
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
        """
        Retrieves an image-caption pair, processes them, and returns as a dictionary.
        """
        # Get the image filename and caption
        image_filename = self.data.iloc[idx]["image"]
        caption = self.data.iloc[idx]["caption"]

        # Construct the full path to the image
        image_path = os.path.join(self.images_folder, image_filename)

        # Check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open the image and convert to RGB
        raw_image = Image.open(image_path).convert("RGB")

        # Preprocess the image
        inputs = self.processor(
            images=raw_image,
            return_tensors="pt"
        )

        # Tokenize the caption and process labels
        caption_input = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        labels = caption_input.input_ids.squeeze(0)

        # Replace padding tokens with -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Add the labels to the inputs
        inputs["labels"] = labels

        # Return the processed data (squeezing batch dimension)
        return {key: val.squeeze(0) for key, val in inputs.items()}


def fine_tune_model(model, processor):
    captions_file = "data/captions.txt"  # Path to captions.txt
    images_folder = "data/images"  # Path to images folder

    try:
        # Create the dataset and dataloader
        dataset = ImageCaptionDataset(captions_file, images_folder, processor)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Set up the optimizer
        optimizer = AdamW(model.parameters(), lr=5e-5)
        model.train()

        print("Fine-tuning the model...")
        for epoch in range(3):  # Number of epochs
            total_loss = 0
            for i, batch in enumerate(dataloader):
                pixel_values = batch["pixel_values"]
                labels = batch["labels"]

                # Prepare the input_ids and replace ignored tokens (-100) with pad token
                input_ids = labels.clone()
                input_ids[input_ids == -100] = processor.tokenizer.pad_token_id

                # Debugging: Check the final inputs to the model
                print(f"[DEBUG] Batch {i + 1}")
                print(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
                print(f"[DEBUG] input_ids shape: {input_ids.shape}")
                print(f"[DEBUG] labels shape: {labels.shape}")

                # Forward pass: Pass the image pixel values, input_ids, and labels to the model
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

                # Compute and accumulate the loss
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader):.4f}")

        # Save the fine-tuned model and processor
        model.save_pretrained("fine_tuned_model")
        processor.save_pretrained("fine_tuned_model")
        print("Fine-tuning complete. Model saved as 'fine_tuned_model'.")
    except Exception as e:
        print(f"Error during fine-tuning: {e}")


# Main execution
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

nltk.download('punkt_tab')
# Store performance metrics globally
performance_metrics = {
    'time_taken': 0,
    'memory_used': 0,
    'bleu_score': 0,
    'rouge_score': {'rouge1': 0},
    'meteor_score': 0
}

captions = {
    'user_caption': "",
    'generated_caption': ""
}


# Set the Tesseract executable path for macOS
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Path for macOS installed via Homebrew

# Load pre-trained BLIP model and processor
print("Loading BLIP model and processor...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

# Global variable to store image path and user caption
image_path = None
user_caption = ""  # Store user caption globally

def calculate_rouge_score(reference_caption, generated_caption):
    # Tokenize both reference and generated captions
    reference_tokens = ' '.join(word_tokenize(reference_caption.lower()))  # Tokenized and lowercase
    generated_tokens = ' '.join(word_tokenize(generated_caption.lower()))  # Tokenized and lowercase
    
    # Calculate ROUGE scores using Rouge Scorer from rouge_score package
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_tokens, generated_tokens)
    
    # Update ROUGE scores
    rouge_scores = {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }
    return rouge_scores

def calculate_meteor_score(reference_caption, generated_caption):
    # Tokenize the reference and generated captions
    reference_tokens = word_tokenize(reference_caption.lower())  # Tokenize and convert to lowercase
    generated_tokens = word_tokenize(generated_caption.lower())  # Tokenize and convert to lowercase

    # Now calculate METEOR score using tokenized captions
    score = meteor_score([reference_tokens], generated_tokens)
    return score  # Returns METEOR score

def calculate_bleu_score(reference_caption, generated_caption):
    # Ensure that both reference_caption and generated_caption are lists of tokens
    if isinstance(reference_caption, list) and isinstance(generated_caption, list):
        reference = [reference_caption]  # BLEU expects a list of lists for references
        candidate = generated_caption  # Candidate is just a list of tokens
        bleu_score = sentence_bleu(reference, candidate)  # Calculate BLEU score
        print(f"Bleu Score: {bleu_score}")
        return bleu_score
    else:
        raise ValueError("Both the reference and generated captions should be lists of tokens.")
    

def generate_caption(image_path, callback, user_caption):
    def task():
        global captions  # Ensure that we're using the global captions dictionary
        try:
            start_time = time.time()  # Start measuring time
            process = psutil.Process()  # Process to measure memory usage

            print(f"Opening image: {image_path}")
            raw_image = Image.open(image_path).convert("RGB")
            print("Image opened successfully.")

            print("Preprocessing image...")
            inputs = processor(raw_image, return_tensors="pt")
            print(f"Preprocessed input: {inputs}")

            print("Generating caption...")
            out = model.generate(**inputs)
            print(f"Generated output: {out}")

            # Decode the output into text
            generated_caption = processor.decode(out[0], skip_special_tokens=True)
            print(f"Generated caption: {generated_caption}")

            # Save the generated caption in the captions dictionary
            captions['generated_caption'] = generated_caption  # Save generated caption

            # Measure time taken and memory usage
            end_time = time.time()
            elapsed_time = end_time - start_time
            memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB

            # Update performance metrics
            performance_metrics['time_taken'] = elapsed_time
            performance_metrics['memory_used'] = memory_usage

            # Execute callback on the main thread
            root.after(0, callback, generated_caption, elapsed_time, memory_usage)

        except Exception as e:
            print(f"Error during caption generation: {e}")
            root.after(0, messagebox.showerror, "Error", "Failed to generate caption.")
    
    thread = threading.Thread(target=task)
    thread.start()


def update_custom_caption(caption_input, caption_label):
    global user_caption
    user_caption = caption_input.get("1.0", "end-1c").strip()  # Get the input text from Text widget
    if user_caption:
        caption_label.config(text=user_caption)
    else:
        messagebox.showerror("Error", "Please enter a caption.")

# Function to save the user caption
def save_user_caption(caption_input):
    global captions
    user_caption = caption_input.get("1.0", "end-1c").strip()  # Save user input to global variable
    if user_caption:
        print(f"User caption: {user_caption}")
        captions['user_caption'] = user_caption
        messagebox.showinfo("Caption Saved", "Your caption has been saved successfully!")
        update_bleu_score()
        update_rouge_score()
        update_meteor_score()
    else:
        messagebox.showerror("Error", "Please enter a valid caption.")

# Function to update caption in the GUI (runs on the main thread)
def update_caption(caption, bleu_score, time_taken, memory_used, caption_label):
    caption_label.config(text=caption)
    show_performance_metrics(bleu_score, time_taken, memory_used)

def update_bleu_score():
    global captions, performance_metrics

    # Tokenize the captions
    user_caption_tokens = word_tokenize(captions['user_caption'].lower())  # Tokenize and convert to lowercase
    generated_caption_tokens = word_tokenize(captions['generated_caption'].lower())  # Tokenize and convert to lowercase

    # Calculate BLEU score
    bleu_score = calculate_bleu_score(user_caption_tokens, generated_caption_tokens)

    # Update the performance metrics with the calculated BLEU score
    performance_metrics['bleu_score'] = bleu_score

    print(f"Updated BLEU Score in performance metrics: {performance_metrics['bleu_score']}")

def update_rouge_score():
    global captions, performance_metrics

    # Calculate ROUGE score
    rouge_scores = calculate_rouge_score(captions['user_caption'], captions['generated_caption'])

    # Update the performance metrics with the calculated ROUGE score
    performance_metrics['rouge_score'] = rouge_scores

    print(f"Updated ROUGE Scores in performance metrics: {performance_metrics['rouge_score']}")

def update_meteor_score():
    global captions, performance_metrics

    # Calculate METEOR score
    meteor_score_value = calculate_meteor_score(captions['user_caption'], captions['generated_caption'])

    # Update the performance metrics with the calculated METEOR score
    performance_metrics['meteor_score'] = meteor_score_value

    print(f"Updated METEOR Score in performance metrics: {performance_metrics['meteor_score']}")

def show_performance_metrics(bleu_score, time_taken, memory_used, rouge_scores, meteor_score):
    def create_metrics_window():
        metrics_window = tk.Toplevel(root)  # Ensure root is passed as parent
        metrics_window.title("Performance Metrics")
        
        # Add the metrics to the window
        time_label = ttk.Label(metrics_window, text=f"Time taken: {time_taken:.4f} seconds", font=("Helvetica", 12))
        time_label.grid(row=0, column=0, padx=10, pady=10)
        
        memory_label = ttk.Label(metrics_window, text=f"Memory used: {memory_used:.2f} MB", font=("Helvetica", 12))
        memory_label.grid(row=1, column=0, padx=10, pady=10)

        bleu_label = ttk.Label(metrics_window, text=f"BLEU score: {bleu_score:.4f}", font=("Helvetica", 12))
        bleu_label.grid(row=2, column=0, padx=10, pady=10)

        # Display ROUGE scores (assuming you pass the individual scores for rouge1, rouge2, and rougeL)
        rouge_label = ttk.Label(metrics_window, text=f"ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-2: {rouge_scores['rouge2']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}", font=("Helvetica", 12))
        rouge_label.grid(row=3, column=0, padx=10, pady=10)

        meteor_label = ttk.Label(metrics_window, text=f"Meteor score: {meteor_score:.4f}", font=("Helvetica", 12))
        meteor_label.grid(row=4, column=0, padx=10, pady=10)
        
        # Add a button to close the window
        close_button = ttk.Button(metrics_window, text="Close", command=metrics_window.destroy)
        close_button.grid(row=5, column=0, pady=10)

    # Ensure this is run in the main thread
    root.after(0, create_metrics_window)

# Function to handle image selection
def select_image(root, caption_label, user_caption_input):
    global image_path  # Declare the global variable here
    print("Opening file dialog...")
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg"), ("Image files", "*.jpeg"), ("Image files", "*.png"), ("Image files", "*.bmp"), ("Image files", "*.tiff")]
    )
    
    if image_path:
        print(f"Image selected: {image_path}")
        show_image(image_path, root)
        generate_button = ttk.Button(root, text="Generate Caption", command=lambda: generate_caption(image_path, lambda caption, bleu_score, time_taken, memory_used: update_caption(caption, bleu_score, time_taken, memory_used, caption_label), user_caption))
        generate_button.grid(row=2, column=0, pady=20, padx=10, sticky="ew")
    else:
        messagebox.showerror("Error", "No image selected.")
    
    return image_path

# Function to display the selected image
def show_image(image_path, root):
    img = Image.open(image_path)
    img.thumbnail((400, 400))  # Resize image
    img_display = ImageTk.PhotoImage(img)
    image_label = tk.Label(root, image=img_display, bd=2, relief="solid")
    image_label.image = img_display  # Keep a reference to the image
    image_label.grid(row=1, column=0, pady=20, sticky="nsew")


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
create_gui()