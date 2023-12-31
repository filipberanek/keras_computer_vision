# Import libraries
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf
import matplotlib.pyplot as plt
import visualkeras
from PIL import ImageFont
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from collections import defaultdict


input_folder_name = r"./input_data/zip_kagglecatsanddogs_5340/PetImages"
output_folder = r"./01_image_classification_from_scratch/outputs"

"""
When working with lots of real-world image data, corrupted images are a common occurence. 
Let's filter out badly-encoded images that do not feature the string "JFIF" in their header.
"""
run_file_cleanup = False
if run_file_cleanup:
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(input_folder_name, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print(f"Deleted {num_skipped} images.")

"""
Lets generate dataset

Dataset is generated using path to images. Folders are categories and we specified
validation split. 

Eventhough images have different sizes, we are resizing them and also specifing number of images
per batch.
"""
image_size = (180, 180)
batch_size = 32

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    input_folder_name,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
print(f"Clases are {train_ds.class_names}")
print(f"There is {len(train_ds.file_paths)} train file")
print(f"There is {len(val_ds.file_paths)} validation file")

"""
Visualize the images on input
"""
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()
"""
Unfortunately we have only few thousands of images, therefore we need to do data augmentation
in order to enhance our dataset. How this augmentation looks like is visualized here
"""
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(factor=0.2),
    layers.RandomContrast(factor = 0.1),
    # You can find more info here: https://www.tensorflow.org/tutorials/images/data_augmentation
    # And here https://keras.io/api/layers/preprocessing_layers/image_augmentation/
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")
plt.show()
# There are to way how to preprocess data
"""
Option 1 - is lazy loading and therefore is more optimal. Its resource efficient, but on the other hand
it doesnt allow you to see data, until they are obtained

inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model
"""

"""
Option 2 - is preprocessed it now, so we can see preprocessed data. 
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))
"""
# Configure dataset for performance
# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


"""
Define model
"""
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

# Instantiate model
model = make_model(input_shape=image_size + (3,), num_classes=2)

"""
Visualize resulted network
"""
keras.utils.plot_model(model,to_file=os.path.join(output_folder, 'model_plot_keras.png'), show_shapes=True, show_layer_names=True)
font = ImageFont.truetype("arial.ttf", 32)
color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'
visualkeras.layered_view(model, to_file = os.path.join(output_folder, 'model_visualkeras.png'), legend=True, font=font, draw_volume=True, color_map=color_map)

"""
Train model
"""
epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(output_folder,"save_at_{epoch}.keras")),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

"""
Run interference
"""

img = keras.utils.load_img(os.path.join(input_folder_name,"Cat/6779.jpg"), target_size=image_size)
plt.figure()
plt.imshow(img)
plt.show()

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")