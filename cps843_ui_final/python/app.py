from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time
import psutil
import pytesseract
import os
import io
import pyttsx3
import tempfile
import shutil
import logging
import nltk
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

user_caption = ""

nltk.download('punkt_tab')

captions = {
    'user_caption': "",
    'generated_caption': ""
}

# Store performance metrics globally
performance_metrics = {
    'time_taken': 0,
    'memory_used': 0,
    'bleu_score': 0,
    'rouge_score': {'rouge1': 0},
    'meteor_score': 0
}

# Set the Tesseract executable path for macOS
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Path for macOS installed via Homebrew

# Load pre-trained BLIP model and processor
logging.debug("Loading BLIP model and processor...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    logging.debug("Model and processor loaded successfully.")
except Exception as e:
    logging.debug(f"Error loading model or processor: {e}")
    exit()

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

# Function to generate a caption and measure performance
def generate_caption(image_path):
    try:
        global captions
        start_time = time.time()  # Start measuring time
        process = psutil.Process()  # Process to measure memory usage

        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)

        # Decode the output into text
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions['generated_caption'] = caption

        # Measure time taken and memory usage
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB

        performance_metrics['time_taken'] = elapsed_time
        performance_metrics['memory_used'] = memory_usage
        
        return caption, elapsed_time, memory_usage

    except Exception as e:
        return None, 0, 0
    
# Endpoint to upload an image and generate a caption
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_path = os.path.join('/tmp', file.filename)
        file.save(image_path)

        caption, time_taken, memory_used = generate_caption(image_path)
        if caption:
            response = {
                'caption': caption,
                'performance': {
                    'time_taken': time_taken,
                    'memory_used': memory_used
                }
            }
            return jsonify(response), 200
        else:
            return jsonify({'error': 'Failed to generate caption'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Endpoint to perform OCR on an image
@app.route('/ocr', methods=['POST'])
def perform_ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_path = os.path.join('/tmp', file.filename)
        file.save(image_path)

        img = Image.open(image_path)
        ocr_text = pytesseract.image_to_string(img)  # Extract text from image
        if ocr_text.strip():
            return jsonify({'ocr_text': ocr_text}), 200
        else:
            return jsonify({'ocr_text': 'No text found in the image'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# Endpoint to download the image
@app.route('/image/<filename>', methods=['GET'])
def download_image(filename):
    file_path = os.path.join('/tmp', filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'File not found'}), 404

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

    return performance_metrics['bleu_score']

def update_rouge_score():
    global captions, performance_metrics

    # Calculate ROUGE score
    rouge_scores = calculate_rouge_score(captions['user_caption'], captions['generated_caption'])

    # Update the performance metrics with the calculated ROUGE score
    performance_metrics['rouge_score'] = rouge_scores

    print(f"Updated ROUGE Scores in performance metrics: {performance_metrics['rouge_score']}")

    return performance_metrics['rouge_score']

def update_meteor_score():
    global captions, performance_metrics

    # Calculate METEOR score
    meteor_score_value = calculate_meteor_score(captions['user_caption'], captions['generated_caption'])

    # Update the performance metrics with the calculated METEOR score
    performance_metrics['meteor_score'] = meteor_score_value

    print(f"Updated METEOR Score in performance metrics: {performance_metrics['meteor_score']}")

    return performance_metrics['meteor_score']


@app.route('/submit_caption', methods=['POST'])
def submit_caption():
    global captions
    global performance_metrics

    try:
        # Get the user caption from the request
        user_caption = request.json.get('user_caption')
        if not user_caption:
            return jsonify({'error': 'No user caption provided'}), 400
        
        captions['user_caption'] = user_caption
        
        # Debugging: Log the user and generated captions
        logging.debug(f"User Caption: {captions['user_caption']}")
        logging.debug(f"Generated Caption: {captions['generated_caption']}")

        # Calculate BLEU, ROUGE, and METEOR scores

        update_bleu_score()
        update_rouge_score()
        update_meteor_score()
        

        # Debugging prints
        logging.debug(f"BLEU Score: {update_bleu_score()}")
        logging.debug(f"ROUGE Scores: {update_rouge_score()}")
        logging.debug(f"METEOR Score: {update_meteor_score()}")

        # Update performance metrics
        performance_metrics['bleu_score'] = update_bleu_score()
        performance_metrics['rouge_score'] = update_rouge_score()
        performance_metrics['meteor_score'] = update_meteor_score()

        # Return the performance metrics
        response = {
            'user_caption': captions['user_caption'],
            'generated_caption': captions['generated_caption'],
            'performance': performance_metrics
        }
        logging.debug(f"Performance Metrics: {performance_metrics}")  # Debug print
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Endpoint to get performance metrics
@app.route('/performance', methods=['GET'])

def get_performance_metrics():
    return jsonify(performance_metrics), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)