from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "model.keras"  # Change this to your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg","bmp"}

# Model output categories
categories = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]


# Function to check allowed file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to preprocess image (resize + normalize)
def preprocess_image(image):
    resized_image = image.resize((224, 224))  # Resize to model input size
    image_array = np.array(resized_image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return (
            jsonify({"error": "Invalid file type. Supported formats: PNG, JPG, JPEG"}),
            415,
        )

    try:
        # Open and preprocess image
        image = Image.open(file)
        processed_image = preprocess_image(image)

        # Run inference
        prediction = model.predict(processed_image)
        predicted_class = categories[
            np.argmax(prediction)
        ]  # Map prediction to class label

        # Generate response
        response = {
            "prediction": predicted_class,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
