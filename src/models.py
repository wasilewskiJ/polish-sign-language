"""
Model definitions for PSL recognition.
"""

import tensorflow as tf
from .config import Config


def build_keras_mlp(input_shape, num_classes):
    """Build Keras MLP model for landmark features (78 → 128 → 64 → classes)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(Config.KERAS_L2_REG),
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(Config.KERAS_DROPOUT),
        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(Config.KERAS_L2_REG),
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(Config.KERAS_DROPOUT),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.KERAS_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape, num_classes):
    """Build lightweight CNN model for raw images (Conv → Conv → Dense)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.CNN_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
