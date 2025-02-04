import torch
from PIL import Image
import os
from flask import Flask, request, jsonify
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
demo_images_path = 'demo_images'
app = Flask(__name__)

def process_images():
    results = {}
    for img_name in os.listdir(demo_images_path):
        img_path = os.path.join(demo_images_path, img_name)
        if img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            img = Image.open(img_path).convert('RGB')
            img = img_transform(img).unsqueeze(0)
            
            logits = parseq(img)
            pred = logits.softmax(-1)
            label, confidence = parseq.tokenizer.decode(pred)
            
            results[img_name] = label[0]
    return results

@app.route('/', methods=['GET'])
def home():
    return "Hey there!"

@app.route('/post', methods=['POST'])
def listen_for_post():
    results = process_images()
    return jsonify(results)

if __name__ == '__main__':
    parseq = torch.hub.load('baudm/parseq', 'crnn', pretrained=True, device="cuda").eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    app.run(host='0.0.0.0', port=5000)
