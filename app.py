from flask import Flask, render_template, request
from model import predict_emotion
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

DB_PATH = "database/users.db"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        image_path TEXT,
        emotion TEXT,
        confidence REAL,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        file = request.files["image"]

        if not file:
            return "Please upload an image."

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        emotion, confidence = predict_emotion(file_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, image_path, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                       (name, file_path, emotion, confidence, timestamp))
        conn.commit()
        conn.close()

        return render_template("result.html", name=name, image_path=file_path,
                               emotion=emotion, confidence=round(confidence*100, 2),
                               timestamp=timestamp)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
