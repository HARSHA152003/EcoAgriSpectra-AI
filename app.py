from flask import Flask, render_template, request, redirect, jsonify, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from joblib import load
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

dmodel = load_model("Supporting Files\Dbest_model.h5")
ddf = pd.read_csv("Supporting Files\class_mapping main.csv")

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def predict_image_class(model, img_path, labels):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_name = labels.loc[labels['Label'] == predicted_class, 'Class_Name'].values[0]
    return class_name

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model
model_path = 'Supporting Files\pbest.pt'  # Update with your actual model path
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/disease-section', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        file_path = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
        else:
            data = request.get_json()
            if 'image' in data:
                image_data = data['image']
                image_data = base64.b64decode(image_data.split(",")[1])
                image = Image.open(BytesIO(image_data))
                filename = "captured_image.jpg"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(file_path)

        predicted_class = predict_image_class(dmodel, file_path, ddf)
        return jsonify({'result': f'Detected disease: {predicted_class}'})

    return render_template('disease.html')

@app.route('/pest-section', methods=['GET', 'POST'])
def pest_detection():
    if request.method == 'POST':
        if 'file' in request.files:  # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
        else:  # Handle base64 image
            data = request.get_json()
            if 'image' in data:
                image_data = data['image']
                # Decode the image
                image_data = image_data.split(",")[1]
                image_data = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_data))
                filename = "captured_image.jpg"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(file_path)

        # Predict using YOLO model
        results = model.predict(source=file_path)
        
        # Get the label and confidence score
        labels_and_scores = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)  # Convert tensor to float
                labels_and_scores.append((label, confidence))
        
        # Save the image with bounding boxes
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        results[0].save(output_path)
        
        if labels_and_scores:
            label, confidence = labels_and_scores[0]
            result = f"Detected pest: {label} with confidence {confidence:.2f}"
        else:
            result = "No pest detected"
        
        processed_image_path = os.path.join('/results', filename)  # Assuming this is the accessible path
        return jsonify({'status': 'success', 'filename': filename, 'result': result, 'processedImage': processed_image_path})

    return render_template('pest.html')

@app.route('/pest-result')
def pest_result():
    filename = request.args.get('filename')
    result = request.args.get('result')
    return render_template('pest_result.html', filename=filename, result=result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

data = pd.read_csv('Supporting Files\encoded_dataset.csv')

# Prepare X (features) and y (target)
X = data[['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_encoded', 'Season_encoded', 'State_encoded']]
y = data['Yield']

# Define a function to train the model
def train_model(X, y):
    model = ExtraTreesRegressor(random_state=42)
    model.fit(X, y)
    return model

# Train the model
model_rf = train_model(X, y)

@app.route('/yield-section', methods=['GET', 'POST'])
def yield_section():
    yield_prediction = None
    
    if request.method == 'POST':
        # Retrieve user inputs
        crop = int(request.form['crop'])
        season = int(request.form['season'])
        state = int(request.form['state'])
        area = float(request.form['area'])
        production = float(request.form['production'])
        rainfall = float(request.form['rainfall'])
        fertilizer = float(request.form['fertilizer'])
        pesticide = float(request.form['pesticide'])
        year = int(request.form['year'])
        
        # Create input array for the model
        input_features = np.array([[year, area, production, rainfall, fertilizer, pesticide, crop, season, state]])
        columns = ['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_encoded', 'Season_encoded', 'State_encoded']
        X_new = pd.DataFrame(input_features, columns=columns)
        
        # Predict using the trained model
        y_pred = model_rf.predict(X_new)
        yield_prediction = y_pred[0]
    
    # Render the template with the predicted yield
    return render_template('yield.html', yield_prediction=yield_prediction)

@app.route('/soil-section', methods=['GET', 'POST'])
def soil():
    total_sum = None

    if request.method == 'POST':
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            pH = float(request.form['pH'])
            EC = float(request.form['EC'])
            OC = float(request.form['OC'])
            S = float(request.form['S'])
            Zn = float(request.form['Zn'])
            Fe = float(request.form['Fe'])
            Cu = float(request.form['Cu'])
            Mn = float(request.form['Mn'])
            B = float(request.form['B'])

            input_data = [N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]

            file_path = "Supporting Files\dataset1.csv"
            data = pd.read_csv(file_path)
            target = 'Output'
            features = data.drop(columns=[target])
            target_data = data[target]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(features, target_data)
            input_list = [input_data]
            feature_names = list(features.columns)
            input_data = pd.DataFrame(input_list, columns=feature_names)
            predictions = model.predict(input_data)
            if predictions == [0]:
                total_sum = "Not Fertile"
            elif predictions == [1]:
                total_sum = "Fertile"
            elif predictions == [2]:
                total_sum = "Highly Fertile"
        except Exception as e:
            total_sum = f"Error: {e}"

    return render_template('soil.html', total_sum=total_sum)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
