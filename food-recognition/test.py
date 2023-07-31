import tensorflow as tf
import os
import pandas as pd

test_folder = "data/val/val_set"

test_csv = "data/annot/val_info.csv"
test_data = pd.read_csv(test_csv)

test_image_to_label = dict(zip(test_data["image_path"], test_data["class_label"]))

def load_and_preprocess_image_test(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Set 'channels' to 1 for grayscale images

    target_height = 128
    target_width = 128
    image = tf.image.resize(image, [target_height, target_width])

    image = image / 255.0
    image_name = os.path.basename(image_path.numpy().decode())
    label = test_image_to_label[image_name]
    return image, label
test_image_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]

test_dataset = tf.data.Dataset.from_tensor_slices(test_image_files)
test_dataset = test_dataset.map(lambda x: tf.py_function(load_and_preprocess_image_test, [x], [tf.float32, tf.int32]))

def set_shapes(image, label):
    image.set_shape((128, 128, 3))  # Set the shape of the image tensor
    label.set_shape(())  # Set the shape of the label tensor
    return image, label
test_dataset = test_dataset.map(set_shapes)

test_dataset = test_dataset.map(lambda image, label: (image, tf.squeeze(label)))

batch_size = 32

test_dataset = test_dataset.batch(batch_size)

model = tf.keras.models.load_model("model-128-ResNet.h5")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

model.evaluate(test_dataset)
