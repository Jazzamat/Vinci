import tensorflow as tf
from VAE import DualInputVAE
from data_preprocessing import load_data  # This should include your data loading and preprocessing functions

def train_vae(spectrogram_paths, image_paths, epochs, batch_size, learning_rate):
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
    train_image_paths  = ['/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/Me And My Broken Heart/cover (64, 64).png']
    train_song_paths = ['/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/7 Years/song.wav']
    epochs = 2000
    batch_size = 32
    learning_rate = 0.0001

    train_vae(train_song_paths, train_image_paths, epochs, batch_size, learning_rate)
