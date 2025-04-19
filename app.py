from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model_tbc (2).h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Konfigurasi upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Konversi ke RGB jika gambar memiliki channel alpha
    img = img.resize((320, 320))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'})
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Preprocess gambar
        processed_image = preprocess_image(file_path)
        
        # Prediksi
        prediction = model.predict(processed_image)
        print("Raw prediction value:", prediction[0][0])  # Log nilai prediksi mentah
        
        # Ubah threshold menjadi lebih tinggi untuk mengurangi false positive
        result = 'TBC' if prediction[0][0] > 0.85 else 'Normal'
        confidence = float(prediction[0][0] if result == 'TBC' else 1 - prediction[0][0])
        
        print(f"Final result: {result} with confidence: {confidence}")  # Log hasil akhir
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'raw_prediction': float(prediction[0][0]),  # Tambahkan nilai prediksi mentah ke response
            'image_path': file_path
        })

if __name__ == '__main__':
    app.run(debug=True) 