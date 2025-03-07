from flask import Flask, request, jsonify, render_template
import torch
import pickle
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from joblib import load  # Use joblib for safer model loading

app = Flask(__name__)

# Load the saved SVM model and class names
try:
    svm_model = load("svm_model.pkl")  # Use joblib instead of pickle
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    if not isinstance(class_names, (list, dict)):
        raise ValueError("class_names.pkl does not contain a valid list or dictionary.")

except Exception as e:
    print("Error loading model files:", e)
    exit()

# Load Vision Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimizes inference on CUDA

vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
vit_model.head = torch.nn.Identity()  # Remove classification head
vit_model.to(device)
vit_model.eval()

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform(image).unsqueeze(0).to(device)

# Extract features from image
def extract_features(image_tensor):
    with torch.no_grad():
        features = vit_model(image_tensor).cpu().numpy()
    return features.squeeze()

# Home page route
@app.route("/")
def index():
    return render_template("index.html")  # Serves the HTML page

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        image = Image.open(file).convert("RGB")
        image_tensor = preprocess_image(image)
        features = extract_features(image_tensor)
        
        # Predict using SVM
        predicted_class = svm_model.predict([features])[0]
        predicted_label = class_names[predicted_class]

        return jsonify({"prediction": predicted_label})
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
