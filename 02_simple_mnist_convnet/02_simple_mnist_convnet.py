import numpy as np
import keras
from keras import layers
import visualkeras
import os

"""
DESCRIPTION

We are about to use downloaded dataset for training. Network that will be trained will be defined from scratch, 
so we can se construction of the network and its visualization. Difference is that, dataset is a bit preprocessed and smaall. 
Network is similar. Its very small as there is no need for huge network, so training is very quick. 
"""

# Define variables 
output_folder = r"./02_simple_mnist_convnet/outputs"

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Observe unique value within dataset
print(np.unique(x_train))

# As its easier for network to adapt on value between 0,1, we rescale it
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices, basically one hot matrix
print(f"Original shape is {y_train.shape}")
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"One hot shape is {y_train.shape}")

# Build model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"), # We used softmax, so our values will be between 0-1, alias percentage.
    ]
)

print(model.summary())
keras.utils.plot_model(model,to_file=os.path.join(output_folder, 'model_plot_keras.png'), show_shapes=True, show_layer_names=True)
visualkeras.layered_view(model, to_file = os.path.join(output_folder, 'model_visualkeras.png'), legend=True, draw_volume=True)

# Train model 
batch_size = 32
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate model 
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])