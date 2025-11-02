from flask import Flask, render_template, request
from model import predict_emotion
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    name = request.form.get("name")
    file = request.files.get("image")

    if not file:
        return "No file uploaded", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    label, confidence = predict_emotion(file_path)

    return render_template(
        "result.html",
        name=name,
        filename=file.filename,
        label=label,
        confidence=confidence,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
