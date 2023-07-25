"""
Generator for GAN model.
"""

from tensorflow import keras
from keras.layers import LeakyReLU

def create_generator(latent_dim: int):
    """
    Create generator model.
    """
    gmodel = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            keras.layers.Dense(8 * 8 * 128),
            keras.layers.Reshape((8, 8, 128)),

            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same",
                                          activation=LeakyReLU(alpha=0.2)),
            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same",
                                          activation=LeakyReLU(alpha=0.2)),
            keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same",
                                          activation=LeakyReLU(alpha=0.2)),
            keras.layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

    return gmodel

if __name__ == '__main__':
    model = create_generator(100)
    model.summary()
