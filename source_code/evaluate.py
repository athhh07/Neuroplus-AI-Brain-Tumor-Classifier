import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "brain_tumor_dataset/Testing",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

model = tf.keras.models.load_model("models/brain_tumor_classifier_model.h5")

loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy * 100, "%")