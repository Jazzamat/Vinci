import tensorflow as tf
import glob
from VAE import DualInputVAE
from data_preprocessing import load_data  # This should include your data loading and preprocessing functions

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



def train_vae(spectrogram_paths, image_paths, epochs, batch_size, learning_rate):
    with tf.device('/GPU:0'):
        # Load and preprocess data
        train_dataset = load_data(spectrogram_paths, image_paths)

        buffer_size = 1000
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

    epochs = 500
    batch_size = 300
    learning_rate = 0.00001

    train_vae(train_song_paths, train_image_paths, epochs, batch_size, learning_rate)
