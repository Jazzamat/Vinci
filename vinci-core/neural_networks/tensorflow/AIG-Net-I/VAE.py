import tensorflow as tf
# Configure TensorFlow to use the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
import glob
from data_preprocessing import load_data 





class DualInputVAE:

    with tf.device('/GPU:0'):
        def __init__(self, spectrogram_shape, image_shape, latent_dim):
            self.spectrogram_shape = spectrogram_shape
            self.image_shape = image_shape
            self.latent_dim = latent_dim

            self.spectrogram_encoder = self._build_encoder(spectrogram_shape, 'spectrogram_encoder')
            self.image_encoder = self._build_encoder(image_shape, 'image_encoder')
            self.decoder = self._build_decoder()
            self.vae = self._build_vae()

        def _build_encoder(self, input_shape, name):
            inputs = Input(shape=input_shape)
            x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
            x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
            x = Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)  # Added layer
            x = Conv2D(256, 3, activation='relu', strides=2, padding='same')(x)  # Added layer
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)  # Increased size
            x = Dense(64, activation='relu')(x)   # Added layer

            mean = Dense(self.latent_dim, name=f'{name}_mean')(x)
            log_var = Dense(self.latent_dim, name=f'{name}_log_var')(x)

            def sampling(args):
                mean, log_var = args
                epsilon = tf.random.normal(shape=(tf.shape(mean)[0], self.latent_dim))
                return mean + tf.exp(0.5 * log_var) * epsilon

            z = Lambda(sampling, name=f'{name}_sampling')([mean, log_var])
            return Model(inputs, [mean, log_var, z], name=name)

        def _build_decoder(self):
            latent_inputs = Input(shape=(self.latent_dim * 2,))
            x = Dense(8 * 8 * 128, activation='relu')(latent_inputs)  # Increased filter size
            x = Reshape((8, 8, 128))(x)
            x = Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)  # Upsample to 16x16
            x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)   # Upsample to 32x32
            x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)   # Upsample to 64x64
            outputs = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)      # Final layer, 64x64x3

            return Model(latent_inputs, outputs, name='decoder')


        def _build_vae(self):
            spectrogram_inputs = Input(shape=self.spectrogram_shape)
            image_inputs = Input(shape=self.image_shape)

            spectrogram_mean, spectrogram_log_var, spectrogram_z = self.spectrogram_encoder(spectrogram_inputs)
            image_mean, image_log_var, image_z = self.image_encoder(image_inputs)

            merged_z = Concatenate()([spectrogram_z, image_z])
            decoder_outputs = self.decoder(merged_z)

            vae = Model([spectrogram_inputs, image_inputs], decoder_outputs, name='dual_input_vae')

            reconstruction_loss = MeanSquaredError()(image_inputs, decoder_outputs)
            kl_loss_spectrogram = -0.5 * tf.reduce_sum(1 + spectrogram_log_var - tf.square(spectrogram_mean) - tf.exp(spectrogram_log_var), axis=-1)
            kl_loss_image = -0.5 * tf.reduce_sum(1 + image_log_var - tf.square(image_mean) - tf.exp(image_log_var), axis=-1)

            total_loss = reconstruction_loss + kl_loss_spectrogram + kl_loss_image
            vae.add_loss(total_loss)

            return vae



def train_vae(spectrogram_paths, image_paths, epochs, batch_size, learning_rate):
    with tf.device('/GPU:0'):
        # Load and preprocess data
        train_dataset = load_data(spectrogram_paths, image_paths)

        buffer_size = 32
        train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size)

        # Initialize the VAE model
        dual_vae = DualInputVAE(spectrogram_shape=(64, 64, 1), image_shape=(64, 64, 3), latent_dim=32)

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Training loop
        for epoch in range(epochs):
            for spectrogram_batch, image_batch in train_dataset:
                with tf.GradientTape() as tape:
                    reconstructed_images = dual_vae.vae([spectrogram_batch, image_batch])
                    total_loss = sum(dual_vae.vae.losses)

                gradients = tape.gradient(total_loss, dual_vae.vae.trainable_variables)
                optimizer.apply_gradients(zip(gradients, dual_vae.vae.trainable_variables))

            print(f'Epoch {epoch+1}, total_loss: {total_loss.numpy()}')


        dual_vae.vae.save_weights('./model_weights.h5')


if __name__ == "__main__":


    # Example file paths and parameters
    #train_image_paths  = ['/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/Me And My Broken Heart/cover (64, 64).png']
    train_image_paths = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/cover (64, 64).png")
    train_song_paths = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/*.wav")

    print(f"Number of songs: {len(train_song_paths)}")
    print(f"Number of images: {len(train_image_paths)}")

    epochs = 2000
    batch_size = 32
    learning_rate = 0.0001

    train_vae(train_song_paths, train_image_paths, epochs, batch_size, learning_rate)
