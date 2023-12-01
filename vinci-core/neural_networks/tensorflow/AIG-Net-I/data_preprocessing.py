import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

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

from PIL import Image

class AudioPreprocessor:

    @staticmethod
    def wav_to_spectrogram(file_path, recalculate=False, n_fft=2048, hop_length=512):
        output_dir = os.path.dirname(file_path)
        base_filename = os.path.basename(file_path)
        spectrogram_path = os.path.join(output_dir, 'spectrogram.png')

        if not recalculate and os.path.exists(spectrogram_path):
            return Image.open(spectrogram_path)

        # Load audio file and convert to spectrogram
        signal, sr = librosa.load(file_path, sr=22050)
        stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)

         # Convert to image, normalize to [0, 255], and resize
        spectrogram_image = Image.fromarray(log_spectrogram)
        min_val = np.min(spectrogram_image)
        max_val = np.max(spectrogram_image)
        spectrogram_image = (spectrogram_image - min_val) / (max_val - min_val) * 255  # Normalize to [0, 255]
        spectrogram_image = spectrogram_image.astype(np.uint8)  # Convert to unsigned byte format
        spectrogram_image = Image.fromarray(spectrogram_image)
        spectrogram_image = spectrogram_image.resize((64, 64))

        # Ensure the image is in the correct mode before saving
        if spectrogram_image.mode != 'L':
            spectrogram_image = spectrogram_image.convert('L')

        # Save the spectrogram
        spectrogram_image.save(spectrogram_path)
        print(f"Saved spectrogram to {spectrogram_path}")
        return spectrogram_image
    # Additional methods if needed

def load_and_preprocess_image(path, target_size=(64, 64)):
    image = Image.open(path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

def load_data(spectrogram_paths, image_paths, recalculate=False):
    spectrograms = []
    images = []

    print("Processing .wav files to spectrograms... ")
    for path in spectrogram_paths:
        spectrogram_image = AudioPreprocessor.wav_to_spectrogram(path, recalculate)
        log_spectrogram_resized = np.array(spectrogram_image)[..., np.newaxis]
        spectrograms.append(log_spectrogram_resized)

    print("Loading images... ")
    for path in image_paths:
        image = load_and_preprocess_image(path)
        images.append(image)

    # Convert lists to tensors and ensure correct dimensions
    spectrograms = np.array(spectrograms)[..., np.newaxis]  # Add channel dimension
    images = np.array(images)

    print("Creating TensorFlow dataset... ")

    return tf.data.Dataset.from_tensor_slices((spectrograms, images))


#Usage:
if __name__ == "__main__":
    spectrogram_output_dir = 'path/to/cache/spectrograms'
    os.makedirs(spectrogram_output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    spectrogram_paths = ['path/to/spectrogram1.wav', 'path/to/spectrogram2.wav', ...]
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
    dataset = load_data(spectrogram_paths, image_paths, spectrogram_output_dir)
