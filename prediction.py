from importlib.resources import path
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from keras.models import load_model

model = None

class_names = ['blonde', 'brunette', 'gingers']

def loadModel():
    model = load_model('models/model.h5')
    print("Model loaded")
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = loadModel()
        
    Img = np.asarray(image.resize((180, 180)))[..., :3]
    img_array = tf.keras.utils.img_to_array(Img)
    img_array = tf.expand_dims(img_array, 0)
    new_model = load_model('models/model.h5')
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prediction = class_names[np.argmax(score)]

    return prediction    
    
    
    
    
    


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image