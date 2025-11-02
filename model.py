from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image

MODEL_NAME = "dima806/facial_emotions_image_detection"

print("Downloading and loading emotion detection model...")
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")


def predict_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax().item()
        label = model.config.id2label[pred_idx]
        confidence = probs[0][pred_idx].item()

    return label, confidence
