import tensorflow as tf
import glob
from model import build_encoder, build_decoder, build_fcnn
from data_preprocessing import load_data  # This should include your data loading and preprocessing functions
import argparse
# Function to train an autoencoder
def train_autoencoder(encoder, decoder, dataset, epochs, optimizer):
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                latent = encoder(batch)
                reconstructed = decoder(latent)
                loss = tf.reduce_mean(tf.square(batch - reconstructed))

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
   

    train_image_paths = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/cover (64, 64).png")
    train_song_paths = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/*.wav")


    print(f"Number of songs: {len(train_song_paths)}")
    print(f"Number of images: {len(train_image_paths)}")


    train_dataset, spectrogram_latent, image_latent = load_data(train_song_paths, train_image_paths, recalculate) # Adjust this according to your data preprocessing


    print("Initializing models")
    # Initialize the models
    spectrogram_encoder = build_encoder(spectrogram_shape, latent_dim)
    spectrogram_decoder = build_decoder(latent_dim, spectrogram_shape)
    image_encoder = build_encoder(image_shape, latent_dim)
    image_decoder = build_decoder(latent_dim, image_shape)
    fcnn = build_fcnn(latent_dim, latent_dim)

    # Training parameters
    epochs = 1000
    batch_size = 100
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Training autoencoders
    print("Training Spectrogram Autoencoder")
    train_autoencoder(spectrogram_encoder, spectrogram_decoder, train_dataset.map(lambda x, y: x), epochs, optimizer)

    print("Training Image Autoencoder")
    train_autoencoder(image_encoder, image_decoder, train_dataset.map(lambda x, y: y), epochs, optimizer)

    # Preparing dataset for FCNN training
    fcnn_dataset = tf.data.Dataset.from_tensor_slices((spectrogram_latent, image_latent)).batch(batch_size)

    # Training FCNN
    print("Training FCNN")
    train_fcnn(fcnn, fcnn_dataset, epochs, optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to generate images from audio.')
    parser.add_argument('--recalculate', action='store_true',
                        help='Recalculate all spectrograms instead of using cached ones')
    args = parser.parse_args()

    main(recalculate=args.recalculate)
