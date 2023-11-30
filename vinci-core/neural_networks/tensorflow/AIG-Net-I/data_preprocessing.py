import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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
    def wav_to_spectrogram(file_path, n_fft=2048, hop_length=512):
        signal, sr = librosa.load(file_path, sr=22050)
        stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)

        spectrogram_image = Image.fromarray(log_spectrogram)
        spectrogram_image = spectrogram_image.resize((64,64))
        log_spectrogram_resized = np.array(spectrogram_image)

        return log_spectrogram_resized[..., np.newaxis] 

    # Additional methods if needed

def load_and_preprocess_image(path, target_size=(64, 64)):
    image = Image.open(path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

def load_data(spectrogram_paths, image_paths):
    spectrograms = []
    images = []

    for path in spectrogram_paths:
        spectrogram = AudioPreprocessor.wav_to_spectrogram(path)
        spectrograms.append(spectrogram)

    for path in image_paths:
        image = load_and_preprocess_image(path)
        images.append(image)

    # Convert lists to tensors and ensure correct dimensions
    spectrograms = np.array(spectrograms)[..., np.newaxis]  # Add channel dimension
    images = np.array(images)

    return tf.data.Dataset.from_tensor_slices((spectrograms, images))

#
if __name__ == "__main__":
    spectrogram_paths = ['path/to/spectrogram1.wav', 'path/to/spectrogram2.wav', ...]
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
    dataset = load_data(spectrogram_paths, image_paths)
