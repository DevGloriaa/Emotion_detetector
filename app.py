from flask import Flask, render_template, request, jsonify
from model import load_model, predict_emotion
import os

app = Flask(__name__)
model = load_model('emotion_model.h5') 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    if image_file:
       
        filepath = os.path.join('database', image_file.filename)
        image_file.save(filepath)
       
        result = predict_emotion(model, filepath)
       
        return jsonify({'emotion': result})
    return jsonify({'error': 'No image uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
