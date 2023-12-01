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
    x = Dense(128, activation='relu')(latent_inputs)
    x = Dense(output_shape[0] * output_shape[1] * output_shape[2] // 4, activation='relu')(x)
    x = Reshape((output_shape[0] // 2, output_shape[1] // 2, output_shape[2]))(x)
    x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
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
