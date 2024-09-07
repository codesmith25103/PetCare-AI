from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from tensorflow.keras.applications.densenet import preprocess_input

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('/home/lata/Desktop/PetCare-AI/Cows.h5')

# Load class names from a text file or predefined list
class_names = []
with open('/home/lata/Desktop/PetCare-AI/class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f]

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image_array = np.array(image)      # Convert to numpy array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocess for DenseNet121
    return image_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pet-image' not in request.files:
        return jsonify({'result': 'No image uploaded'}), 400

    file = request.files['pet-image']
    symptoms = request.form.get('symptoms', '')

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            image_array = preprocess_image(image)
            prediction = model.predict(image_array)
            class_idx = np.argmax(prediction, axis=1)[0]
            
            # Ensure class_names are correctly mapped
            predicted_class = class_names[class_idx]
            
            result = f'Predicted class: {predicted_class}'
            
            if symptoms:
                result += f'\nSymptoms provided: {symptoms}'

            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'result': f'Error processing image: {str(e)}'}), 500

    return jsonify({'result': 'No image file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
