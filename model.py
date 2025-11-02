from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch


MODEL_NAME = "tae898/emotion-detection-facial"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

def predict_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class].item()

    label = model.config.id2label[predicted_class]
    return label, round(confidence * 100, 2)
