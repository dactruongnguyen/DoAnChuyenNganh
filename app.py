from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load the CNN model
cnn_model = None
try:
    cnn_model = tf.keras.models.load_model("cucumber_disease_CNN_model.keras")
except Exception as e:
    print(f"CNN model loading failed: {e}")

# Load the VGG model
vgg_model = None
try:
    vgg_model = tf.keras.models.load_model("cucumber_disease_VGG16_model.keras")
except Exception as e:
    print(f"VGG model loading failed: {e}")

# Load the ResNet model
resnet_model = None
try:
    resnet_model = tf.keras.models.load_model("Classification Cucumber ResNet50.keras")
except Exception as e:
    print(f"ResNet model loading failed: {e}")

# Define the class names
class_names = [
    "Anthracnose", "Bacterial Wilt", "Belly Rot", "Downy Mildew",
    "Fresh Cucumber", "Fresh Leaf", "Gummy Stem Blight", "Pythium Fruit Rot"
]

def preprocess_image(img):
    """Preprocess the image for model prediction."""
    # Mở hình ảnh với Pillow và chuyển đổi sang RGB
    image = Image.open(img).convert("RGB")
    
    # Resize hình ảnh
    image = image.resize((224, 224))  # Thay đổi kích thước ảnh về 224x224
    
    # Chuyển đổi hình ảnh thành mảng numpy và chuẩn hóa
    input_arr = np.array(image) / 255.0
    
    # Đảm bảo rằng hình ảnh có ba kênh (RGB)
    if input_arr.shape[-1] != 3:
        # Thêm kênh nếu cần
        input_arr = np.concatenate([input_arr] * 3, axis=-1)
    
    # Thay đổi hình ảnh thành định dạng (1, 224, 224, 3) để phù hợp với đầu vào mô hình
    input_arr = np.expand_dims(input_arr, axis=0)
    
    return input_arr


def cnn_model_prediction(input_arr):
    """Predict the class of the input image using the CNN model."""
    if cnn_model is None:
        raise Exception("CNN model is not loaded.")
    predictions = cnn_model.predict(input_arr)
    return predictions

def vgg_model_prediction(input_arr):
    """Predict the class of the input image using the VGG model."""
    if vgg_model is None:
        raise Exception("VGG model is not loaded.")
    predictions = vgg_model.predict(input_arr)
    return predictions

def resnet_model_prediction(input_arr):
    """Predict the class of the input image using the ResNet model."""
    if resnet_model is None:
        raise Exception("ResNet model is not loaded.")
    predictions = resnet_model.predict(input_arr)
    return predictions

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    model_type = request.form.get('model', 'cnn')  # Get the model type from form data

    try:
        # Perform prediction
        input_arr = preprocess_image(file)
        if model_type == 'resnet':
            if resnet_model is None:
                return jsonify({"error": "ResNet model is not available"}), 500
            predictions = resnet_model_prediction(input_arr)
            class_index = np.argmax(predictions)
        elif model_type == 'vgg':
            if vgg_model is None:
                return jsonify({"error": "VGG model is not available"}), 500
            predictions = vgg_model_prediction(input_arr)
            class_index = np.argmax(predictions)
        else:
            if cnn_model is None:
                return jsonify({"error": "CNN model is not available"}), 500
            predictions = cnn_model_prediction(input_arr)
            class_index = np.argmax(predictions)
        
        class_name = class_names[class_index]
        confidence = np.max(predictions) * 100  # Convert to percentage

        return jsonify({
            "disease": class_name,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
