import os
import cv2
import numpy as np

def load_data(data_dir, IMG_WIDTH, IMG_HEIGHT, NUM_CATEGORIES):
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        print(f"Loading category: {category_dir}")
        for filename in os.listdir(category_dir):
            if filename.endswith(".ppm") or filename.endswith(".jpg"): 
                img_path = os.path.join(category_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(category)

    return np.array(images), np.array(labels)


data_dir = "gtsrb"
IMG_HEIGHT = 30
IMG_WIDTH = 30
NUM_CATEGORIES = 43

# Load data
images, labels = load_data(data_dir, IMG_WIDTH, IMG_HEIGHT, NUM_CATEGORIES)

# Flatten images into row vectors of 2700 values
reshaped_images = images.reshape(images.shape[0], -1)

# Print shapes to confirm
print(f"Images shape: {reshaped_images.shape}, Labels shape: {labels.shape}")


