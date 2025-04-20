from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model_tbc.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    try:
        # Use keras.preprocessing.image to match notebook
        img = image.load_img(image_path, target_size=(320, 320), interpolation='lanczos')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        print(f"Preprocessed image shape: {img_array.shape}")
        print(f"Preprocessed image min/max: {img_array.min():.6f}/{img_array.max():.6f}")
        print(f"Preprocessed image mean: {img_array.mean():.6f}")

        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess image
        processed_image = preprocess_image(file_path)

        # Predict
        prediction = model.predict(processed_image)
        prob = prediction[0][0]  # Probability of TBC
        print(f"Raw prediction (TBC probability): {prob:.6f}")

        # Apply threshold
        classes = ['Normal', 'TBC']
        result = classes[1] if prob >= 0.5 else classes[0]
        confidence = prob if result == 'TBC' else 1 - prob

        print(f"Result: {result}, Confidence: {confidence:.6f}")

        return jsonify({
            'result': result,
            'confidence': float(confidence),
            'tbc_probability': float(prob),
            'normal_probability': float(1 - prob),
            'image_path': file_path
        })

if __name__ == '__main__':
    app.run(debug=True)