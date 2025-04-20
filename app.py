from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model_tbc.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess image (match training preprocessing!)
def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(320, 320))  # Default interpolation
        img_array = image.img_to_array(img)

        # Apply samplewise center + std normalization
        img_array -= np.mean(img_array)
        img_array /= (np.std(img_array) + 1e-7)

        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)
    prob = prediction[0][0]

    classes = ['Normal', 'TBC']
    result = classes[1] if prob >= 0.5 else classes[0]
    confidence = prob if result == 'TBC' else 1 - prob

    return jsonify({
        'result': result,
        'confidence': float(confidence),
        'tbc_probability': float(prob),
        'normal_probability': float(1 - prob),
        'image_path': file_path
    })

if __name__ == '__main__':
    app.run(debug=True)
