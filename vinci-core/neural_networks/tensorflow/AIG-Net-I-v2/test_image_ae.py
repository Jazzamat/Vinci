import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_encoder, build_decoder
from PIL import Image

def load_and_preprocess_image(path, target_size=(64, 64)):
    image = Image.open(path).convert('RGB')  # Convert to RGB if needed
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return image

def test_image_autoencoder(encoder_weights, decoder_weights, image_path, image_shape, latent_dim):
    # Load the encoder and decoder models
    image_encoder = build_encoder(image_shape, latent_dim)
    image_encoder.load_weights(encoder_weights)

    image_decoder = build_decoder(latent_dim, image_shape)
    image_decoder.load_weights(decoder_weights)

    # Load and preprocess the image
    original_image = load_and_preprocess_image(image_path, image_shape[:2])
    original_image = original_image[np.newaxis, ...]  # Add batch dimension

    # Encode and decode the image
    encoded_image = image_encoder.predict(original_image)
    decoded_image = image_decoder.predict(encoded_image)

    # Plot original and reconstructed images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image[0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(decoded_image[0], cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    plt.show()

# Specify the model weights and image path
encoder_weights = 'weights/image_encoder_weights.h5'
decoder_weights = 'weights/image_decoder_weights.h5'
image_path = '/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/Circles/cover (64, 64).png'
image_shape = (64, 64, 3)  # Adjust as per your model
latent_dim = 32  # Adjust as per your model

# Test the image autoencoder
test_image_autoencoder(encoder_weights, decoder_weights, image_path, image_shape, latent_dim)
