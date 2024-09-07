from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pet-image' not in request.files:
        return jsonify({'result': 'No image uploaded'}), 400

    file = request.files['pet-image']

    if file:
        # For simplicity, let's assume the model expects a numpy array with image data
        # In practice, you need to preprocess the image and extract features for your model
        from PIL import Image
        import numpy as np

        image = Image.open(file)
        # Example processing, replace with your actual model requirements
        image_array = np.array(image).flatten().reshape(1, -1)  # Flatten and reshape
        prediction = model.predict(image_array)

        return jsonify({'result': f'Predicted class: {prediction[0]}'})

    return jsonify({'result': 'No image file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
