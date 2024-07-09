import os
import joblib
from sklearn import datasets
import numpy as np
from PIL import Image

# Load the model
model = joblib.load('model.joblib')

# Function to preprocess and predict on an image
def predict_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((8, 8), Image.ANTIALIAS)
    image = np.array(image)
    image = 16 - (np.mean(image, axis=2) / 255 * 16)
    image = image.flatten().reshape(1, -1)
    return model.predict(image)

# Directory for input images
input_dir = '/input_images'
output_file = '/output/predictions.txt'

with open(output_file, 'w') as f:
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            prediction = predict_image(os.path.join(input_dir, filename))
            f.write(f'{filename}: {prediction[0]}\n')
