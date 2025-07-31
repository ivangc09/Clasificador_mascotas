from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

def preprocess_image(file, target_size=(224, 224)):
    img = load_img(file, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)