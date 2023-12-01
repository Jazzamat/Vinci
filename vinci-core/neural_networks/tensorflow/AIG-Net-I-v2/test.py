import tensorflow as tf
from model import build_encoder, build_decoder, build_fcnn
import numpy as np
from data_preprocessing import wav_to_spectrogram
import matplotlib.pyplot as plt

def load_models(spectrogram_encoder_weights, image_decoder_weights, fcnn_weights, spectrogram_shape, image_shape, latent_dim):
    spectrogram_encoder = build_encoder(spectrogram_shape, latent_dim)
    spectrogram_encoder.load_weights(spectrogram_encoder_weights)

    image_decoder = build_decoder(latent_dim, image_shape)
    image_decoder.load_weights(image_decoder_weights)

    fcnn = build_fcnn(latent_dim, latent_dim)
    fcnn.load_weights(fcnn_weights)

    return spectrogram_encoder, image_decoder, fcnn

def generate_album_cover(song_path, spectrogram_encoder, image_decoder, fcnn):
    spectrogram = wav_to_spectrogram(song_path)  # Ensure this matches the training preprocessing
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]  # Reshape for the model

    # Encode the spectrogram to latent space
    spectrogram_latent = spectrogram_encoder.predict(spectrogram)

    # Use FCNN to map spectrogram latent to image latent space
    image_latent = fcnn.predict(spectrogram_latent)

    # Decode to get the album cover
    album_cover = image_decoder.predict(image_latent)

    return album_cover

# Paths to model weightss
spectrogram_encoder_weights = 'weights/spectrogram_encoder_weights.h5'
image_decoder_weights = 'weights/image_decoder_weights.h5'
fcnn_weights = 'weights/fcnn_weights.h5'

# Model parameters
spectrogram_shape = (64, 64, 3)
image_shape = (64, 64, 3)
latent_dim = 32

# Load models
spectrogram_encoder, image_decoder, fcnn = load_models(spectrogram_encoder_weights, image_decoder_weights, fcnn_weights, spectrogram_shape, image_shape, latent_dim)

# Generate an album cover for a new song
new_song_path = '/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/Anti-Hero/song.wav'
album_cover = generate_album_cover(new_song_path, spectrogram_encoder, image_decoder, fcnn)

# Display or save the album cover
if album_cover.ndim > 3:
    album_cover = album_cover[0]  # Assuming batch dimension is present

# Display the album cover
plt.imshow(album_cover, cmap='gray')  # Adjust cmap if your image is not grayscale
plt.axis('off')
plt.show()
