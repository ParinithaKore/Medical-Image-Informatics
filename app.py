
""""

# Load your model
#model = load_model('C:/Users/Deepak/Downloads/medical_image_model.h5')

# app.py
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('C:/Users/Deepak/Downloads/medical_image_model.h5')  # Load your trained model

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Load and preprocess the image
    img = image.load_img(file, target_size=(128, 128))  # Adjust size as per model requirements
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    disease_class = "Anomaly Detected" if prediction[0][0] > 0.5 else "Normal"  # Adjust threshold as needed

    return jsonify({"result": disease_class})

if __name__ == "__main__":
    app.run(debug=True)

"""
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        img = image.load_img(BytesIO(file.read()), target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        return jsonify({'class': str(predicted_class[0])})  # Replace with actual class mapping

    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)
