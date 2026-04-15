import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

def show_images(folder, label):
    plt.figure(figsize=(8, 4))
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for i in range(4):
        img = load_img(os.path.join(folder, random.choice(files)),
                       target_size=(224, 224))
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.suptitle(label)
    plt.show()