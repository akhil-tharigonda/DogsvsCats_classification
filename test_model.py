import tensorflow as tf
import numpy as np
import cv2

# Load Trained Model
model = tf.keras.models.load_model("cat_vs_dog_cnn_model.h5")

# Load Image
image_path = "C:/Users/akhil/OneDrive/Desktop/Internship/CatVsDogs/dataset/test/cats/7999.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (64, 64))  # Resize to match model input shape
image = np.expand_dims(image, axis=0) / 255.0

# Predict
prediction = model.predict(image)
label = "Dog" if prediction > 0.5 else "Cat"
print(f"Predicted Label: {label}")
