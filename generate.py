"""
Generate some images using the trained generator model.
"""

import argparse

from tensorflow import keras
from matplotlib import pyplot as plt

def generate(checkpoint_path: str) -> None:
    """
    Generate some images using the trained generator model.
    :param checkpoint_path: The path to the checkpoint file.
    :return: None.
    """
    # Load generator from h5 file
    generator = keras.models.load_model(checkpoint_path)
    latent_dim = generator.input_shape[1]

    # Generate images
    noise = keras.backend.random_normal(shape=(16, latent_dim))
    generated_images = generator(noise, training=False)

    # Plot images
    plt.figure(figsize=(4, 4))

    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, :])
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images using the trained generator.")
    parser.add_argument("checkpoint_path", type=str, help="The path to the checkpoint file.")

    args = parser.parse_args()

    generate(args.checkpoint_path)
