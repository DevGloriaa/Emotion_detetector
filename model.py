def load_model(model_path):
    from tensorflow.keras.models import load_model
    return load_model(model_path)

def predict_emotion(model, image_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = image.load_img(image_path, target_size=(48,48), color_mode="grayscale")
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    return emotions[np.argmax(predictions)]
