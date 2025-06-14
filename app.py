from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    with open('boston_house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please train and save the model first.")
    model = None

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input features"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data from request
        data = request.get_json()
        
        # Extract features in the correct order for Boston Housing dataset
        features = [
            float(data['crim']),     # Crime rate
            float(data['zn']),       # Residential land zoned
            float(data['indus']),    # Industrial business acres
            float(data['chas']),     # Charles River (0 or 1)
            float(data['nox']),      # NOx concentration
            float(data['rm']),       # Average rooms per dwelling
            float(data['age']),      # Age of units built before 1940
            float(data['dis']),      # Distance to employment centers
            float(data['rad']),      # Highway accessibility
            float(data['tax']),      # Property tax rate
            float(data['ptratio']),  # Pupil-teacher ratio
            float(data['b']),        # Black population proportion
            float(data['lstat'])     # Lower status population
        ]
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Ensure prediction is not negative
        prediction = max(0, prediction)
        
        return jsonify({
            'prediction': round(prediction, 2),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)