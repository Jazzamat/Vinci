from VAE import DualInputVAE
from data_preprocessing import AudioPreprocessor
import numpy as np
import matplotlib.pyplot as plt

def generate_album_cover(song_path, dual_vae, dummy_image_latent_dim=32):
    # Preprocess the song to get its spectrogram
    spectrogram = AudioPreprocessor.wav_to_spectrogram(song_path)  # Ensure this matches the training preprocessing

    # Encode the spectrogram
    spectrogram_mean, spectrogram_log_var, spectrogram_z = dual_vae.spectrogram_encoder.predict(spectrogram[np.newaxis, ...])

    # Generate a dummy latent representation for the image
    dummy_image_z = np.zeros((1, dummy_image_latent_dim))

    # Merge latent representations
    merged_z = np.concatenate([spectrogram_z, dummy_image_z], axis=-1)

    # Decode to get the album cover
    album_cover = dual_vae.decoder.predict(merged_z)

    return album_cover

def load_model(weights_path, spectrogram_shape, image_shape, latent_dim):
    # Initialize the VAE model with the same architecture parameters as used during training
    dual_vae = DualInputVAE(spectrogram_shape=spectrogram_shape, 
                            image_shape=image_shape, 
                            latent_dim=latent_dim)

    # Load the saved weights
    dual_vae.vae.load_weights(weights_path)

    return dual_vae

# Specify the path to your saved weights and the model parameters
weights_path = './model_weights.h5'
spectrogram_shape = (64, 64, 1)  # Example shape, adjust as per your model
image_shape = (64, 64, 3)        # Adjust as per your model
latent_dim = 32                  # Adjust as per your model

# Load the model
dual_vae = load_model(weights_path, spectrogram_shape, image_shape, latent_dim)

# Now you can use dual_vae for inference, testing, etc.


# Generate an album cover for a new song
new_song_path = '/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/7 Years/song.wav'
album_cover = generate_album_cover(new_song_path, dual_vae)

# Display or save the album cover
if album_cover.ndim > 3:
    album_cover = album_cover.reshape(album_cover.shape[1:])

# Display the album cover
plt.imshow(album_cover, cmap='gray')  # Use cmap='gray' if it's a grayscale image
plt.axis('off')  # Turn off axis numbers and labels
plt.show()

