# app.py
import os
import io
import base64
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "trained_models/emotion_model.h5"
DB_PATH = "database/predictions.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app = Flask(__name__)
detector = MTCNN()
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    print("Model not found at", MODEL_PATH)

# Database setup
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    person_name = Column(String, default="anonymous")
    image_path = Column(String)
    prediction = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # adjust if model differs

def preprocess_face(img: Image.Image):
    img = img.convert('L').resize((48, 48))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 48, 48, 1)
    return arr

def detect_face_and_predict(pil_image):
    img_array = np.asarray(pil_image)
    # detect faces
    res = detector.detect_faces(img_array)
    if not res:
        return None, None
    # choose first face
    x, y, w, h = res[0]['box']
    x, y = max(0, x), max(0, y)
    face = pil_image.crop((x, y, x+w, y+h))
    pre = preprocess_face(face)
    preds = model.predict(pre)[0]
    idx = np.argmax(preds)
    return EMOTIONS[idx], float(preds[idx])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # supports file form field 'file' and optional 'name'
    file = request.files.get('file')
    person_name = request.form.get('name', 'anonymous')
    if not file:
        return jsonify({"error": "no file uploaded"}), 400
    image = Image.open(file.stream).convert('RGB')
    emotion, score = detect_face_and_predict(image)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{person_name}_{timestamp}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    # save to db
    db = SessionLocal()
    pred = Prediction(person_name=person_name, image_path=path, prediction=f"{emotion}:{score:.4f}")
    db.add(pred)
    db.commit()
    db.close()

    return jsonify({"emotion": emotion, "score": float(score), "image": path})

@app.route("/webcam", methods=["POST"])
def webcam():
    # expects JSON: {"image": "data:image/jpeg;base64,...", "name": "..." }
    data = request.get_json()
    img_b64 = data.get("image", "")
    person_name = data.get("name", "anonymous")
    if not img_b64:
        return jsonify({"error": "no image data"}), 400
    header, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
    binary = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(binary)).convert('RGB')

    emotion, score = detect_face_and_predict(image)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{person_name}_{timestamp}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path)

    db = SessionLocal()
    pred = Prediction(person_name=person_name, image_path=path, prediction=f"{emotion}:{score:.4f}")
    db.add(pred)
    db.commit()
    db.close()

    return jsonify({"emotion": emotion, "score": float(score), "image": path})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/history")
def history():
    db = SessionLocal()
    rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(100).all()
    db.close()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "name": r.person_name,
            "image": r.image_path,
            "prediction": r.prediction,
            "created_at": r.created_at.isoformat()
        })
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)
