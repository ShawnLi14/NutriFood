import tensorflow as tf
import os
import pandas as pd
import random

train_folder = "data/train/train_set"
test_folder = "data/val/val_set"

train_csv = "data/annot/train_info.csv"
train_data = pd.read_csv(train_csv)

from tensorflow.keras.applications import ResNet50

train_image_to_label = dict(zip(train_data["image_path"], train_data["class_label"]))


test_csv = "data/annot/val_info.csv"
test_data = pd.read_csv(test_csv)

test_image_to_label = dict(zip(test_data["image_path"], test_data["class_label"]))

image_size = 128

def load_and_preprocess_image_train(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Set 'channels' to 1 for grayscale images

    target_height = image_size
    target_width = image_size
    image = tf.image.resize(image, [target_height, target_width])

    image = image / 255.0
    image_name = os.path.basename(image_path.numpy().decode())
    label = train_image_to_label[image_name]
    return image, label

def load_and_preprocess_image_test(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Set 'channels' to 1 for grayscale images

    target_height = image_size
    target_width = image_size
    image = tf.image.resize(image, [target_height, target_width])

    image = image / 255.0
    image_name = os.path.basename(image_path.numpy().decode())
    label = test_image_to_label[image_name]
    return image, label

def augment_image(image, label):
    # Randomly apply horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)

    # Randomly apply rotation (range in radians)
    rotation_angle = tf.random.uniform((), minval=-0.2, maxval=0.2)
    image = tf.image.rot90(image, k=tf.cast(rotation_angle / (2 * 3.1416) * 4, tf.int32))

    # Randomly apply brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Randomly apply zoom
    if tf.random.uniform(()) > 0.5:
        zoom_factor = tf.random.uniform((), minval=0.9, maxval=1.1)
        new_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * zoom_factor, tf.int32)
        image = tf.image.resize(image, new_size)
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)

    return image, label

train_image_files = [os.path.join(train_folder, file) for file in os.listdir(train_folder)]
test_image_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]

random.shuffle(train_image_files)
random.shuffle(test_image_files)

train_dataset = tf.data.Dataset.from_tensor_slices(train_image_files)
train_dataset = train_dataset.map(lambda x: tf.py_function(load_and_preprocess_image_train, [x], [tf.float32, tf.int32]))

test_dataset = tf.data.Dataset.from_tensor_slices(test_image_files)
test_dataset = test_dataset.map(lambda x: tf.py_function(load_and_preprocess_image_test, [x], [tf.float32, tf.int32]))

def set_shapes(image, label):
    image.set_shape((image_size, image_size, 3))  # Set the shape of the image tensor
    label.set_shape(())  # Set the shape of the label tensor
    return image, label

train_dataset = train_dataset.map(set_shapes)
test_dataset = test_dataset.map(set_shapes)

# Unpack the label tensor and remove the extra dimension
train_dataset = train_dataset.map(lambda image, label: (image, tf.squeeze(label)))
test_dataset = test_dataset.map(lambda image, label: (image, tf.squeeze(label)))

train_dataset = train_dataset.map(augment_image)

batch_size = 32
shuffle_buffer_size = 32

train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.shuffle(shuffle_buffer_size)

test_dataset = test_dataset.batch(batch_size)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(251, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

epochs = 10

model.fit(train_dataset, epochs=epochs)

model.evaluate(test_dataset)

model.save("model-128-ResNet-2.h5")
