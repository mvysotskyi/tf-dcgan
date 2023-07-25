"""
Callbacks for the training process.
"""

from tensorflow import keras

class SaveEpochCallback(keras.callbacks.Callback):
    """
    DCGAN callback.
    """
    def __init__(self, model: keras.Model, latent_dim: int):
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Callback on epoch end.
        """
        # Sample random points in the latent space
        random_latent_vectors = keras.backend.random_normal(shape=(3, self.latent_dim))

        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 255).numpy().astype("uint8")

        # Save images
        for i in range(0, generated_images.shape[0]):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(f"generated_img_{epoch}_{i}.png")

        # Save models
        # self.model.generator.save(f"h5s\\generator_{epoch}.h5")
        # self.model.discriminator.save(f"h5s\\discriminator_{epoch}.h5")
