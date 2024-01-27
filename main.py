import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras import (
    Sequential, layers,
    optimizers, losses)

# Constants
_TRAIN_RATIO: float = 0.8
_SPLIT_RATIO: float = 0.1
_IMG_SIZE: int = 224

# Let's get malaria dataset.
dataset = tfds.load('malaria', as_supervised=True,
                    shuffle_files=True, split=['train'])


# We need to split dataset into training, validation and testing.
def splits(data):
    size_dataset = len(data)

    train = data.take(int(_TRAIN_RATIO * size_dataset))
    val = data.skip(int(_TRAIN_RATIO * size_dataset)) \
        .take(int(_SPLIT_RATIO * size_dataset))
    test = data.skip(int((_TRAIN_RATIO + _SPLIT_RATIO) * size_dataset)) \
        .take(int(_SPLIT_RATIO * size_dataset))

    return train, val, test


train_dataset, validation_dataset, test_dataset = splits(dataset[0])


# We need to process images for our model.
def resize_rescale(image, label):
    return tf.image.resize(image, (_IMG_SIZE, _IMG_SIZE, 3)) / 255.0, label


train_dataset = train_dataset.map(resize_rescale)
validation_dataset = validation_dataset(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

# Prepare datasets for easier use, prefetch them and batch, also shuffle train and validation.
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# Now we can create the model.
model = Sequential([
    layers.InputLayer(input_shape=(_IMG_SIZE, _IMG_SIZE, 3)),

    layers.Conv2D(filters=6, kernel_size=3, strides=1, padding='valid', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=2, strides=2),

    layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=2, strides=2),

    layers.Flatten(),

    layers.Dense(100, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid'),
])

# Let's compile it.
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.1),
    loss=losses.BinaryCrossentropy(),
    metrics='accuracy'
)

# Now let's train it!
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50,
    verbose=1
)

# Model evaluation.
test_dataset = test_dataset.batch(1)
model.evaluate(test_dataset)
