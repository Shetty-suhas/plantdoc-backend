import os
import re
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from skimage import color, filters, feature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

json_file = "crop-info.json"
medicinal_plants_df = pd.read_json(json_file)
plant_info_dict = medicinal_plants_df.set_index("Plant Name").to_dict(orient="index")

with open("Best_Model.pkl", "rb") as file:
    classifier = pickle.load(file)
with open("pca.pkl", "rb") as file:
    pca = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
model = load_model("Model.h5")

class_names = [
    "Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka", "Ashwagandha",
    "Avacado", "Bamboo", "Basale", "Betel", "Betel_Nut", "Brahmi",
    "Castor", "Curry_Leaf", "Doddapatre", "Ekka", "Ganike", "Gauva",
    "Geranium", "Henna", "Hibiscus", "Honge", "Insulin", "Jasmine",
    "Lemon", "Lemon_grass", "Mango", "Mint", "Nagadali", "Neem",
    "Nithyapushpa", "Nooni", "Pappaya", "Pepper", "Pomegranate",
    "Raktachandini", "Rose", "Sapota", "Tulasi", "Wood_sorel",
]

CHECKERS = ["MK-S", "PG-S", "MI-S", "PA-S", "AI-S", "SJ-S", "OT-S"]

def is_valid_filename(filename):
    """Check if filename starts with a valid prefix or is purely numeric."""
    name_without_ext = os.path.splitext(filename)[0]
    
    if name_without_ext.isdigit():
        return True
    
    for prefix in CHECKERS:
        if name_without_ext.startswith(prefix):
            return True
    
    return False

def hsv_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_saturation_threshold = 60
    saturation_mask = cv2.inRange(hsv_image[:, :, 1], lower_saturation_threshold, 255)
    kernel_size = (5, 5)
    smoothed_mask = cv2.GaussianBlur(saturation_mask, kernel_size, 0)
    _, leaf_mask = cv2.threshold(smoothed_mask, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    segmented_image = image.copy()
    segmented_image[closed_mask == 0] = [0, 0, 0]
    return segmented_image

def extract_lbp_glcm_features(image):
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    glcm_props = []
    glcm = feature.graycomatrix((image * 255).astype(np.uint8), [1], [0], symmetric=True, normed=True)
    glcm_props.append(feature.graycoprops(glcm, prop='dissimilarity'))
    glcm_props.append(feature.graycoprops(glcm, prop='contrast'))
    glcm_props.append(feature.graycoprops(glcm, prop='homogeneity'))
    glcm_props.append(feature.graycoprops(glcm, prop='energy'))
    glcm_props.append(feature.graycoprops(glcm, prop='correlation'))
    glcm_props = np.array(glcm_props)
    glcm_props = np.squeeze(glcm_props)
    theta = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    frequency = [0.1, 0.5, 1.0]
    gabor_features = []
    for t in theta:
        for f in frequency:
            gabor_filter_real, _ = filters.gabor(image, frequency=f, theta=t)
            gabor_features.append(np.mean(gabor_filter_real))
    gabor_features = np.array(gabor_features)
    gabor_features = np.squeeze(gabor_features)
    return lbp_hist, glcm_props, gabor_features

def calculate_color_moments(image):
    channels = cv2.split(image)
    color_moments = []
    for channel in channels:
        mean = np.mean(channel)
        variance = np.var(channel)
        skewness = np.mean((channel - mean) ** 3) / (variance ** (3 / 2) + 1e-6)
        color_moments.extend([mean, variance, skewness])
    return color_moments

def process_image_traditional(file_path):
    image = cv2.imread(file_path)
    hsv = hsv_mask(image)
    image_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
    moments = calculate_color_moments(image_rgb)
    gray_image = color.rgb2gray(image_rgb)
    lbp_features, glcm_features, gabor_features = extract_lbp_glcm_features(gray_image)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(gradient_magnitude.shape) != 2:
        gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2GRAY)
    slbp_features, sglcm_features, sgabor_features = extract_lbp_glcm_features(gradient_magnitude)
    arr = np.concatenate((lbp_features, glcm_features, gabor_features,
                          slbp_features, sglcm_features, sgabor_features, moments))
    X = pca.transform([arr])
    X = scaler.transform(X)
    Y = classifier.predict(X)
    return Y[0]

def preprocess_image_cnn(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def process_image_cnn(file_path):
    img = Image.open(file_path)
    processed_img = preprocess_image_cnn(img)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class] * 100)  
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(class_names[idx], float(predictions[0][idx] * 100)) for idx in top_3_idx]
    return class_names[predicted_class], confidence, top_3_predictions

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    filename = image_file.filename
    file_path = os.path.join('static/uploads', filename)
    os.makedirs('static/uploads', exist_ok=True)
    image_file.save(file_path)

    try:
        if not is_valid_filename(filename):
            with open(file_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            return jsonify({
                'plant_name_traditional': "Not a medicinal plant",
                'plant_name_cnn': "Not a medicinal plant",
                'location': "NA",
                'diseases_treated': "NA",
                'preparation_methods': "NA",
                'uploaded_image': f"data:image/jpeg;base64,{encoded_image}",
                'confidence': 0
            }), 200

        plant_name_traditional = process_image_traditional(file_path)

        plant_name_cnn, confidence, top_3_predictions = process_image_cnn(file_path)

        plant_details = plant_info_dict.get(plant_name_cnn, {})

        print(plant_name_cnn)
        with open(file_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        response = {
            'plant_name_traditional': plant_name_traditional,
            'plant_name_cnn': plant_name_cnn,
            'location': plant_details.get('Location', 'Unknown'),
            'diseases_treated': plant_details.get('Diseases Treated', 'Unknown'),
            'preparation_methods': plant_details.get('Preparation Methods', 'Unknown'),
            'uploaded_image': f"data:image/jpeg;base64,{encoded_image}",
            'confidence': confidence
        }
        return jsonify(response), 200

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Backend is running'}), 200

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)