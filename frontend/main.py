
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

# Path to the directory containing the model file
#path = r"frontend"

# Load the trained model
model = load_model('front.h5')

# Define a function to preprocess input features
def preprocess_features(feature1, feature2, feature3):
    # Convert input features to floating-point numbers
    feature1 = float(feature1)
    feature2 = float(feature2)
    feature3 = float(feature3)
    # Preprocess the features (if needed)
    # For example, normalization, scaling, etc.
    return np.array([[feature1, feature2, feature3]])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        feature1 = request.form['feature1']
        feature2 = request.form['feature2']
        feature3 = request.form['feature3']

        # Preprocess the input features
        input_features = preprocess_features(feature1, feature2, feature3)

        # Make prediction
        prediction = model.predict(input_features)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Return the predicted class
        return str(predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
