from flask import Flask, render_template, request, jsonify
from model import load_model, predict_emotion
import os

app = Flask(__name__)
model = load_model("resnet34")  # Load pre-trained model

# Ensure database folder exists
if not os.path.exists("database"):
    os.makedirs("database")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    name = request.form.get('name')
    image_file = request.files.get('image')

    if image_file and name:
        filepath = os.path.join('database', image_file.filename)
        image_file.save(filepath)

        # Predict emotion
        emotion = predict_emotion(model, filepath)

        # Return JSON (or later store to a database)
        return jsonify({
            'name': name,
            'emotion': emotion,
            'filename': image_file.filename
        })
    return jsonify({'error': 'Name or image missing'})

if __name__ == '__main__':
    app.run(debug=True)
