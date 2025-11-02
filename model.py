from fer_pytorch import get_pretrained_model
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load pre-trained PyTorch emotion model
def load_model(model_name="resnet34"):
    model = get_pretrained_model(model_name)
    model.eval()  # Set model to evaluation mode
    return model

# Predict emotion from an image path
def predict_emotion(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img = Image.open(image_path).convert("L")
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output,1)
    
    emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
    return emotions[pred.item()]
