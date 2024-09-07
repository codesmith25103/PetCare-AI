from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import model_from_json
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
with open('Hen_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_from_json(model_dict['architecture'])
    model.set_weights(model_dict['weights'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pet-image' not in request.files:
        return jsonify({'result': 'No image uploaded'}), 400

    file = request.files['pet-image']

    if file:
        # Convert file to an image
        image = Image.open(file.stream)
        
        # Resize and preprocess the image
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = preprocess_input(image_array)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction[0])

        return jsonify({'result': f'Predicted class: {predicted_class}'})

    return jsonify({'result': 'No image file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
