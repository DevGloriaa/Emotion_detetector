from model import train_model


csv_file = 'dataset/fer2013.csv'
model_path = 'emotion_model.pt'


train_model(csv_file, model_path, epochs=3)
