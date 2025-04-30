# 🩺 TBC Detection using Convolutional Neural Network (CNN)

This project aims to detect **Tuberculosis (TBC)** from chest X-ray images using a **Convolutional Neural Network (CNN)**. By leveraging deep learning techniques, the model can learn patterns in X-ray images and classify whether a patient is likely infected with TBC or not.

---

## 🧠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn (for data visualization)
- Scikit-learn (for evaluation metrics)

---

## 🖼️ Dataset

The dataset consists of **chest X-ray images**, categorized into two classes:

- **Positive** – Patients diagnosed with Tuberculosis
- **Negative** – Patients not diagnosed with Tuberculosis

Example public datasets:

- [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

---

## 🚀 Installation & Usage Guide

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For macOS/Linux:
source .venv/bin/activate
# For Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

Required package versions:
```
flask==2
werkzeug==2.0.3
tensorflow==2.13.0
pillow==9.0.0
numpy==1.23.5
gunicorn==20.1.0
```

### 3. Model Preparation
- Ensure the model file `model_tbc.h5` exists in the project root directory
- The model must be compatible with the version of TensorFlow being used

### 4. Running the Application

#### Development Mode (Local)
```bash
# Run Flask development server
python app.py
```
The application will run on `http://localhost:5000`

### 5. Accessing the Application
- Open your browser and access `http://localhost:5000`
- Upload a chest X-ray image for TBC detection
- The system will provide a prediction result

### 6. Project Structure
```
tbc-detection-app/
├── app.py             # Main Flask application file
├── model_tbc.h5       # Trained TensorFlow model
├── requirements.txt   # List of dependencies
├── static/            # Folder for static files (CSS, JS, etc.)
└── templates/         # Folder for HTML templates
```
---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Contributors

- [Alfian Kafilah Baits](https://github.com/alfiannnas)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.