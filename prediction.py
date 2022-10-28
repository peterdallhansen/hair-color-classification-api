from io import BytesIO
from tkinter import Image
import numpy as np
import PIL
import tensorflow as tf

import cv2

import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras.models import load_model
from keras.engine.training import Model
import numpy as np

img_height, img_width = 180, 180

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocess(image: Image.Image):
    img = tf.keras.utils.load_img(image , target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def load_model():
    new_model = load_model('models/model.h5')
    return new_model
    
_model_ = load_model()

def predict(image: np.array):
    predictions = _model_.predict(image)
    score = tf.nn.softmax(predictions[0])
    return score


