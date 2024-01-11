import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf  # For tf.data
import matplotlib.pyplot as plt
import keras
import os
from keras import layers
from keras.applications import EfficientNetB0

"""
DESCRIPTION

We are about to use downloaded dataset for training. We also use predefined and pretrained network. We use this network
and uptrain/finetune it to our dataset. Difference is that network has been trained on very huge dataset and we used this one only
with difference, that head of network will be set to a different number of output labels and we finetune this network. 
"""


# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224
BATCH_SIZE = 32
output_folder = r"./03_image_classification_via_pretrained_network_finetuning/outputs"
no_of_epochs = 10

# Load dataset
dataset_name = "stanford_dogs"
# Here are listed all available datasets https://www.tensorflow.org/datasets/catalog/overview#all_datasets
# This is official webpage https://github.com/tensorflow/datasets
# You can download any of available dataset
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes

# We have to resize images into our wished image size
# Image size must match input size in the network
size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

def format_label(label):
    string_label = label_info.int2str(label)
    return string_label.split("-")[1]


# Visualize 9 images from dataset
label_info = ds_info.features["label"]
plt.figure()
for i, (image, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("{}".format(format_label(label)))
    plt.axis("off")
plt.show()

# Data augmentation as we have only small dataset
img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]


def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images

plt.figure()
for image, label in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(np.expand_dims(image.numpy(), axis=0))
        aug_img = np.array(aug_img)
        plt.imshow(aug_img[0].astype("uint8"))
        plt.title("{}".format(format_label(label)))
        plt.axis("off")
plt.show()



# One-hot / categorical encoding
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def input_preprocess_test(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


# Prepare datasets
ds_train = ds_train.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)

# This will train model from scratch
model = EfficientNetB0(
    include_top=True,
    weights=None,
    classes=NUM_CLASSES,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

hist = model.fit(ds_train, epochs=no_of_epochs, validation_data=ds_test)

def plot_hist(hist, filename):
    plt.figure()
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.show()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close("all")


plot_hist(hist, "learning_from_scratch.png")

# Transfer learning of model from pre-trained weights
"""
Basically we take original pretrained model, cut top layer, freeze original model with its weights. 
With these changes we use previous model as feature extractor and then we add our layers, that will do 
classification. We train only added layers.
"""
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Here we cut top layer
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
model = build_model(num_classes=NUM_CLASSES)

hist = model.fit(ds_train, epochs=no_of_epochs, validation_data=ds_test)
plot_hist(hist, "learning_with_predefined_network_transfer_learning_freezed.png")


# Transfer learning of model from pre-trained weights
"""
Basically we take original pretrained model, cut top layer, unfreeze original model with its weights. 
With these changes we use previous model as feature extractor and then we add our layers, that will do 
classification. We train added layers with 20 last layers of predefined model.
"""

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Here we cut top layer
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
model = build_model(num_classes=NUM_CLASSES)

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


unfreeze_model(model)

hist = model.fit(ds_train, epochs=no_of_epochs, validation_data=ds_test)
plot_hist(hist, "learning_with_predefined_network_transfer_learning_unfreezed_20.png")