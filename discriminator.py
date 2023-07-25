"""
Discriminator model for GAN.
"""

from tensorflow import keras
from keras.layers import LeakyReLU

def create_discriminator():
    """
    Create discriminator model.
    """
    dmodel = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),

            keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same",
                                activation=LeakyReLU(alpha=0.2)),
            keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same",
                                activation=LeakyReLU(alpha=0.2)),
            keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same",
                                activation=LeakyReLU(alpha=0.2)),

            keras.layers.Flatten(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    return dmodel

if __name__ == '__main__':
    model = create_discriminator()
    model.summary()
