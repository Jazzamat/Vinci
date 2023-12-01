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



import librosa
from PIL import Image
import numpy as np

def wav_to_spectrogram(file_path, recalculate=True, n_fft=2048, hop_length=512, target_shape=(64, 64, 3)):
    output_dir = os.path.dirname(file_path)
    spectrogram_path = os.path.join(output_dir, 'spectrogram.png')

    if not recalculate and os.path.exists(spectrogram_path):
        return np.array(Image.open(spectrogram_path))

    # Load audio file
    signal, sr = librosa.load(file_path, sr=22050)

    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # Get the magnitude and phase of the STFT
    magnitude, phase = librosa.magphase(stft)

    # Compute the Mel-scaled spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(S=magnitude**2, sr=sr)

    # Normalize each component
    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    magnitude = normalize(librosa.amplitude_to_db(magnitude))
    phase = normalize(np.angle(stft))
    mel = normalize(librosa.power_to_db(mel_spectrogram))

    # Resize each component
    magnitude_img = Image.fromarray(np.uint8(magnitude * 255)).resize(target_shape[:2])
    phase_img = Image.fromarray(np.uint8(phase * 255)).resize(target_shape[:2])
    mel_img = Image.fromarray(np.uint8(mel * 255)).resize(target_shape[:2])

    # Merge to form an RGB image
    spectrogram_image_rgb = Image.merge("RGB", [magnitude_img, phase_img, mel_img])

    # Save the RGB spectrogram
    spectrogram_image_rgb.save(spectrogram_path)
    print(f"Saved spectrogram to {spectrogram_path}")

    return np.array(spectrogram_image_rgb)


    # Additional methods if needed

def load_and_preprocess_image(path, target_size=(64, 64)):
    image = Image.open(path).convert('RGB')  # Convert to RGB if needed
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return image


def load_data(spectrogram_paths, image_paths, recalculate=False):
    spectrograms = []
    images = []

    print("Processing .wav files to spectrograms...")
    for path in spectrogram_paths:
        spectrogram_image = wav_to_spectrogram(path, recalculate)
        spectrograms.append(spectrogram_image / 255.0)  # Normalize and append

    print("Loading images...")
    for path in image_paths:
        image = load_and_preprocess_image(path)
        images.append(image)  # Images are already normalized in load_and_preprocess_image

    # Convert lists to arrays
    spectrograms = np.array(spectrograms)
    images = np.array(images)

    print("Creating TensorFlow dataset...")
    return tf.data.Dataset.from_tensor_slices((spectrograms, images))



#Usage:
if __name__ == "__main__":
    spectrogram_output_dir = 'path/to/cache/spectrograms'
    os.makedirs(spectrogram_output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    spectrogram_paths = ['path/to/spectrogram1.wav', 'path/to/spectrogram2.wav', ...]
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
    dataset = load_data(spectrogram_paths, image_paths, spectrogram_output_dir)
