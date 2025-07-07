import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)

# Absolute paths
CNN_MODEL_PATH = r"C:\Users\Karthik K\OneDrive\Documents\crop_soil_prediction\model\soil_cnn_model.h5"
YOLO_MODEL_PATH = r"C:\Users\Karthik K\OneDrive\Documents\crop_soil_prediction\model\best.pt"
CSV_PATH = r"C:\Users\Karthik K\OneDrive\Documents\crop_soil_prediction\data\yield_df(AutoRecovered).csv"

# Soil image map
SOIL_IMAGES = {
    "sandy": os.path.join("static", "soil_images", "Sandy.jpg"),
    "clay": os.path.join("static", "soil_images", "Clay.jpg"),
    "loamy": os.path.join("static", "soil_images", "Loamy.jpg"),
    "clay skeletal": os.path.join("static", "soil_images", "ClaySkeletal.jpg")
}
soil_classes = list(SOIL_IMAGES.keys())

# Load models and data
cnn_model = load_model(CNN_MODEL_PATH)
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
df["Item"] = df["Item"].astype(str).str.strip().str.lower()
df["Soil Type"] = df["Soil Type"].astype(str).str.strip().str.lower()
df["State"] = df["State"].astype(str).str.strip().str.lower()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_model(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image.")
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), IMG_SIZE)
        return np.expand_dims(img / 255.0, axis=0)
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

def get_soil_image_path(soil_type):
    if pd.isna(soil_type):
        return None
    for soil in soil_type.split(","):
        key = soil.strip().lower()
        if key in SOIL_IMAGES:
            return SOIL_IMAGES[key]
    return None

def get_crop_details(crop_name):
    crop_name = crop_name.strip().lower()
    crop_data = df[df["Item"] == crop_name]
    if crop_data.empty:
        return None

    soil_type = crop_data.iloc[0]["Soil Type"]
    avg_rainfall = crop_data["Avg Rainfall (mm)"].mean()
    avg_temp = crop_data["Avg Temperature (°C)"].mean()
    states = crop_data["State"].unique()

    predicted_soil_type = "Not available"
    soil_image_web_path = None
    image_path = get_soil_image_path(soil_type)

    if image_path:
        full_path = os.path.abspath(image_path)
        if os.path.exists(full_path):
            img_array = preprocess_image_for_model(full_path)
            if img_array is not None:
                try:
                    pred = cnn_model.predict(img_array, verbose=0)
                    predicted_soil_type = soil_classes[np.argmax(pred)].capitalize()
                    soil_image_web_path = image_path.replace("\\", "/")
                except Exception as e:
                    print(f"CNN prediction error: {e}")
            else:
                print(f"Failed preprocessing for image: {image_path}")
        else:
            print(f"Soil image not found: {full_path}")

    return {
        "crop": crop_name.capitalize(),
        "soil": soil_type.capitalize(),
        "avg_rain": f"{avg_rainfall:.2f} mm",
        "avg_temp": f"{avg_temp:.2f}°C",
        "states": ", ".join([s.capitalize() for s in states]),
        "predicted_soil": predicted_soil_type,
        "soil_image": soil_image_web_path
    }

def detect_crop_from_image(image_path):
    if yolo_model is None:
        print("YOLO model not loaded.")
        return None
    try:
        results = yolo_model(image_path)
        boxes = results[0].boxes
        if boxes is not None and boxes.cls.numel() > 0:
            confs = boxes.conf
            top_index = int(np.argmax(confs.numpy()))
            class_id = int(boxes.cls[top_index])
            return yolo_model.names[class_id].lower()
    except Exception as e:
        print(f"YOLO detection error: {e}")
    return None

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not allowed_file(file.filename):
            return render_template("index.html", error="Invalid or no file selected.")

        filename = file.filename.replace(" ", "_")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        crop = detect_crop_from_image(filepath)
        if not crop:
            return render_template("index.html", error="No crop detected.")

        details = get_crop_details(crop)
        if not details:
            return render_template("index.html", error="No data found for this crop.")

        html_path = filepath.replace("\\", "/")
        return render_template("index.html", image=html_path, **details)

    return render_template("index.html")

# Run Flask app
if __name__ == "__main__":
    app.run(debug=False, port=5000)







