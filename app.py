import torch
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from multiprocessing.dummy import Pool as ThreadPool
from strhub.data.module import SceneTextDataModule

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and transformation

def decode_and_process_image(base64_string):
    """Decode a Base64 image and process it using the model."""
    try:
        # Decode the Base64 string
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform and move image to GPU
        img_tensor = img_transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            logits = parseq(img_tensor)
            pred = logits.softmax(-1)
            label, _ = parseq.tokenizer.decode(pred)

        return label[0]  # Return recognized text
    except Exception as e:
        return f"Error processing image: {str(e)}"

@app.route('/', methods=['GET'])
def home():
    return "Hey there!"

@app.route('/fetchme', methods=['POST'])
def fetchme():
    """Receives Base64 images, processes them, and returns formatted JSON response."""
    try:
        # Extract Base64 image array from request
        data = request.get_json()
        base64_images = data.get("mybase46_images", [])

        if not base64_images:
            return jsonify({"error": "No images provided"}), 400

        # Process images in parallel using threads
        with ThreadPool(13) as pool:
            results = pool.map(decode_and_process_image, base64_images)

        # Format response according to the required structure
        response = {
            "id": "",  # You can assign a real ID here if needed
            "modelID": "morocco",
            "solution": {str(i): results[i] for i in range(len(results))},  # Convert to dict with numbered keys
            "status": "solved",
            "url": ""
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    parseq = torch.hub.load('baudm/parseq', 'crnn', pretrained=True, device=device).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    app.run(host='0.0.0.0', port=10000)
