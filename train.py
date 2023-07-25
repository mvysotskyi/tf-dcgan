"""
Train the model.
"""

from tensorflow import keras

from dcgan import DCGAN
from callbacks import SaveEpochCallback

from generator import create_generator
from discriminator import create_discriminator


if __name__ == '__main__':
    # Use float16
    # tf.keras.backend.set_floatx('float16')

    # Create discriminator and generator
    discriminator = create_discriminator()
    generator = create_generator(128)

    # Load discriminator and generator from h5 files
    # discriminator = keras.models.load_model("h5s\\discriminator_51.h5")
    # generator = keras.models.load_model("h5s\\generator_51.h5")

    dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=128)

    dcgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss_fn=keras.losses.BinaryCrossentropy()
    )

    train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    dataset = train_generator.flow_from_directory(
        directory="data",
        target_size=(64, 64),
        batch_size=16,
        class_mode=None
    )

    save_epoch_callback = SaveEpochCallback(dcgan, 128)
    dcgan.fit(dataset, epochs=10, callbacks=[save_epoch_callback])
