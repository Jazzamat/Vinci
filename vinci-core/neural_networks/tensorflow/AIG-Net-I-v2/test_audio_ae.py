import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_encoder, build_decoder
from data_preprocessing import wav_to_spectrogram

def test_spectrogram_autoencoder(encoder_weights, decoder_weights, audio_file_path, spectrogram_shape, latent_dim):
    # Load the encoder and decoder models
    spectrogram_encoder = build_encoder(spectrogram_shape, latent_dim)
    spectrogram_encoder.load_weights(encoder_weights)

    spectrogram_decoder = build_decoder(latent_dim, spectrogram_shape)
    spectrogram_decoder.load_weights(decoder_weights)

    # Convert the audio file to a spectrogram
    spectrogram_image = wav_to_spectrogram(audio_file_path, recalculate=True)
    spectrogram_image = spectrogram_image / 255.0  # Normalize to [0, 1]
    spectrogram_image = spectrogram_image[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    # Encode and decode the spectrogram
    encoded_spectrogram = spectrogram_encoder.predict(spectrogram_image)
    decoded_spectrogram = spectrogram_decoder.predict(encoded_spectrogram)

    # Plot original and reconstructed spectrograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(spectrogram_image[0, :, :, 0], cmap='gray')
    axes[0].set_title('Original Spectrogram')
    axes[0].axis('off')

    axes[1].imshow(decoded_spectrogram[0, :, :, 0], cmap='gray')
    axes[1].set_title('Reconstructed Spectrogram')
    axes[1].axis('off')

    plt.show()

# Specify the model weights and audio file path
encoder_weights = 'weights/spectrogram_encoder_weights.h5'
decoder_weights = 'weights/spectrogram_decoder_weights.h5'
audio_file_path = '/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/#BrooklynBloodPop! (Slowed & Reverb Edit)/song.wav'  # Update this path
spectrogram_shape = (64, 64, 3)  # Adjust as per your model
latent_dim = 32  # Adjust as per your model

# Test the spectrogram autoencoder with an audio file
test_spectrogram_autoencoder(encoder_weights, decoder_weights, audio_file_path, spectrogram_shape, latent_dim)
