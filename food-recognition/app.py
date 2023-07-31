import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import requests
from PIL import Image

app = Flask(__name__)

label_to_class = {}
txt_file = "data/annot/class_list.txt"
with open(txt_file, 'r') as file:
    for line in file:
        class_index, class_name = line.strip().split(' ')
        class_index = int(class_index)
        label_to_class[class_index] = class_name

# Load your trained TensorFlow model
model = tf.keras.models.load_model('model-128-ResNet.h5')

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']
    image_filename = image.filename
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))  # Save the uploaded image
    image_array = preprocess_image(image)
    prediction = run_inference(image_array)
    # Process the prediction as per your specific model and task

    food_name = prediction
    food_description = "Is the predicted food in the image entered"
    nutrition_facts = {
        "Calories": 200,
        "Protein": 10,
        "Carbohydrates": 30,
        "Fat": 5,
    }
    api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
    query = food_name
    response = requests.get(api_url + query, headers={'X-Api-Key': 'pyKQvf0sXMD21VVo42YVjQ==J6FE6NlsOoiHHiJu'})
    if response.status_code == requests.codes.ok:
        jsonOut = json.loads(response.text)
        print(jsonOut['items'])
        nutrition_facts = jsonOut['items'][0]
    else:
        print("Error:", response.status_code, response.text)

    return render_template('upload.html', name=food_name, description=food_description, nutrition=nutrition_facts, image=image_filename)

def preprocess_image(image):
    # Convert the FileStorage object to a PIL Image object
    img = Image.open(image)

    # Resize and convert the image to an array
    img = img.resize((128, 128))
    image_array = np.array(img)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    return image_array

def run_inference(image_array):
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction[0])  # Get the index of the class with the highest probability
    class_name = label_to_class[class_index]  # Map the class index to the class name using the dictionary
    return class_name

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
