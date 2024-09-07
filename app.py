from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import model_from_json
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)

# # Load models for different pet types
models = {}
for pet_type in ['hen', 'cat', 'cow', 'dog']:
    with open(f'{pet_type}_model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_from_json(model_dict['architecture'])
        model.set_weights(model_dict['weights'])
        models[pet_type] = model

# Load the pre-trained model
# with open('Hen_model.pkl', 'rb') as f:
#     model_dict = pickle.load(f)
#     model = model_from_json(model_dict['architecture'])
#     model.set_weights(model_dict['weights'])

['AvianPox', 'FowlCholera', 'MarekDisease', 'Normal']
# ['Feline_Dermatophytosis', 'Mange', 'NormalCat', 'feline_acne']
dict_cat={
    0:"Feline Dermatophytosis",
    1:"Mange",
    2:"Healthy Cat",
    3:"Feline Acne"
}
dict_hen={
    0:"AvianPox",
    1:"FowlCholera",
    2:"MarekDisease",
    3:"Normal"
}
dict_cow={
    
0:'Actinomycosis Lumpy Jaw ',
1: 'Bovine Papillomatosis', 
2:'foot and mouth disease', 
3:'healthy', 
4:'lumpy skin dsease'
}
dict_dog={
    0:'Conjunctivitis1',1: 'Mange',2: 'Ringworm',3: 'Ticks',4: 'healthy'
}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pet-image' not in request.files:
        return jsonify({'result': 'No image uploaded'}), 400

    if 'pet-type' not in request.form:
        return jsonify({'result': 'No pet type selected'}), 400

    pet_type = request.form['pet-type']
    file = request.files['pet-image']

    if pet_type not in models:
        return jsonify({'result': 'Invalid pet type selected'}), 400

    model = models[pet_type]

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
        ans=""
        print(pet_type)
        if pet_type=="cat":
            return jsonify({'result': f'Predicted class: {dict_cat[predicted_class]}'})
        if pet_type=="hen":
            return jsonify({'result': f'Predicted class: {dict_hen[predicted_class]}'})
        if pet_type=="dog":
            return jsonify({'result': f'Predicted class: {dict_dog[predicted_class]}'})
        if pet_type=="cow":
            return jsonify({'result': f'Predicted class: {dict_cow[predicted_class]}'})
        
        return jsonify({'result': f'Predicted class: {predicted_class}'})

    return jsonify({'result': 'No image file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
