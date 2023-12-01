import tensorflow as tf
import glob
from model import build_encoder, build_decoder, build_fcnn
from data_preprocessing import load_data  # This should include your data loading and preprocessing functions
import argparse
# Function to train an autoencoder
def train_autoencoder(encoder, decoder, dataset, epochs, optimizer, num_channels):
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                batch_float = tf.cast(batch, tf.float32)
                
               
                latent = encoder(batch_float)
                reconstructed = decoder(latent)
                loss = tf.reduce_mean(tf.square(batch_float - reconstructed))

            gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))

        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')




# Function to train the FCNN
def train_fcnn(fcnn, dataset, epochs, optimizer):
    for epoch in range(epochs):
        for spectrogram_latent, image_latent in dataset:
            with tf.GradientTape() as tape:
                predicted_image_latent = fcnn(spectrogram_latent)
                loss = tf.reduce_mean(tf.square(image_latent - predicted_image_latent))

            gradients = tape.gradient(loss, fcnn.trainable_variables)
            optimizer.apply_gradients(zip(gradients, fcnn.trainable_variables))
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')


def main(recalculate):
    # Define the shape of your spectrogram data and the latent dimension for the autoencoders
    spectrogram_shape = (64, 64, 3) # Example shape, adjust as necessary
    image_shape = (64, 64, 3) # Adjust if needed
    latent_dim = 32 # Example latent dimension, adjust as necessary
    
    # Training parameters
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
    
    train_image_paths = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/cover (64, 64).png")
    train_song_paths = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/*.wav")

    print(f"Number of songs: {len(train_song_paths)}")
    print(f"Number of images: {len(train_image_paths)}")

    # Load data (and recalculate spectrograms if specified)
    train_dataset = load_data(train_song_paths, train_image_paths, recalculate)
    train_dataset = train_dataset.batch(batch_size) # Make sure to batch your dataset

    print("Initializing models")
    # Initialize the models
    
    #SPECTOGRAM AUTOENCODER
    spectrogram_encoder = build_encoder(spectrogram_shape, latent_dim)
    spectrogram_decoder = build_decoder(latent_dim, spectrogram_shape)

    #IMAGE AUTOENCODER
    image_encoder = build_encoder(image_shape, latent_dim)
    image_decoder = build_decoder(latent_dim, image_shape)
    fcnn = build_fcnn(latent_dim, latent_dim)

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Building the optimizer with model variables
    model_variables = spectrogram_encoder.trainable_variables + spectrogram_decoder.trainable_variables + image_encoder.trainable_variables + image_decoder.trainable_variables + fcnn.trainable_variables
    optimizer.apply_gradients(zip([tf.zeros_like(v) for v in model_variables], model_variables))


    # Training autoencoders
    print("Training Spectrogram Autoencoder")
    train_autoencoder(spectrogram_encoder, spectrogram_decoder, train_dataset.map(lambda x, y: x), epochs, optimizer, num_channels=1)

    print("Training Image Autoencoder")
    train_autoencoder(image_encoder, image_decoder, train_dataset.map(lambda x, y: y), epochs, optimizer, num_channels=3)

    # Extract latent representations after training the autoencoders
    spectrogram_latent = spectrogram_encoder.predict(train_dataset.map(lambda x, y: x))
    image_latent = image_encoder.predict(train_dataset.map(lambda x, y: y))

    # Preparing dataset for FCNN training
    fcnn_dataset = tf.data.Dataset.from_tensor_slices((spectrogram_latent, image_latent)).batch(batch_size)

    # Training FCNN
    print("Training FCNN")
    train_fcnn(fcnn, fcnn_dataset, epochs, optimizer)


    # Save the weights of the models
    spectrogram_encoder.save_weights('weights/spectrogram_encoder_weights.h5')
    spectrogram_decoder.save_weights('weights/spectrogram_decoder_weights.h5')
    image_encoder.save_weights('weights/image_encoder_weights.h5')
    image_decoder.save_weights('weights/image_decoder_weights.h5')
    fcnn.save_weights('weights/fcnn_weights.h5')

    print("Training completed. Model weights saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to generate images from audio.')
    parser.add_argument('--recalculate', action='store_true',
                        help='Recalculate all spectrograms instead of using cached ones')
    args = parser.parse_args()

    main(recalculate=args.recalculate)


