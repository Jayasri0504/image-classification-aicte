import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("waste_classifier.keras")

# Class names (IMPORTANT: must match folder names exactly)
class_names = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']

# Load image
img_path = "napkin.jpg"   # Put any test image in project folder
img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted Class:", predicted_class)