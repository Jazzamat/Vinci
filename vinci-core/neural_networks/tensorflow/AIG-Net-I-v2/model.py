import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model

def build_encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    latent = Dense(latent_dim)(x)
    return Model(inputs, latent, name='encoder')

def build_decoder(latent_dim, output_shape):
    latent_inputs = Input(shape=(latent_dim,))
    # Start with a dense layer to get the correct number of units to reshape
    x = Dense(8 * 8 * 64, activation='relu')(latent_inputs)
    # Reshape to the starting shape for transposed convolutions
    x = Reshape((8, 8, 64))(x)

    # Upsample to the next stage
    x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)  # Upsample to 16x16
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)  # Upsample to 32x32

    # If your original images are 64x64, you add another upsampling layer
    x = Conv2DTranspose(16, 3, activation='relu', strides=2, padding='same')(x)  # Upsample to 64x64

    # Final layer to output the image with the same number of channels as the input
    outputs = Conv2DTranspose(output_shape[2], 3, activation='sigmoid', padding='same')(x)

    return Model(latent_inputs, outputs, name='decoder')


def build_fcnn(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(output_dim)(x)
    return Model(inputs, outputs, name='fcnn')

# Example Usage
latent_dim = 32  # Example latent dimension
spectrogram_shape = (64, 64, 1)  # Replace with actual spectrogram shape
image_shape = (64, 64, 3)  # Replace with actual image shape

spectrogram_encoder = build_encoder(spectrogram_shape, latent_dim)
spectrogram_decoder = build_decoder(latent_dim, spectrogram_shape)

image_encoder = build_encoder(image_shape, latent_dim)
image_decoder = build_decoder(latent_dim, image_shape)

fcnn = build_fcnn(latent_dim, latent_dim)  # Maps spectrogram latent space to image latent space
