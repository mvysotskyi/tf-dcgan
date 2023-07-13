"""
Train the model.
"""

import tensorflow as tf
from tensorflow import keras

from dcgan import DCGAN
from generator import create_generator
from discriminator import create_discriminator

class DCGANCallback(keras.callbacks.Callback):
    """
    DCGAN callback.
    """
    def __init__(self, model: DCGAN, latent_dim: int):
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Callback on epoch end.
        """
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(3, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)

        # Save images
        for i in range(generated_images.shape[0]):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(f"generated_img_{epoch}_{i}.png")


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    generator = create_generator(128)
    discriminator = create_discriminator()
    dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=128)

    dcgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss_fn=keras.losses.BinaryCrossentropy()
    )

    dataset = keras.preprocessing.image_dataset_from_directory(
        "data\\img_align_celeba",
        label_mode=None,
        image_size=(64, 64),
        batch_size=32,
        # shuffle=True
    )

    dataset = dataset.map(lambda x: x / 255.0)
    # Get 30000 images from dataset
    # dataset = dataset.take(320)

    callback = DCGANCallback(dcgan, 128)
    dcgan.fit(dataset, epochs=5, callbacks=[callback], batch_size=32)

    # Save models
    generator.save("generator.h5")
    discriminator.save("discriminator.h5")

