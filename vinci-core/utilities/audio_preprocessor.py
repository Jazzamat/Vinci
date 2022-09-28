# audio preporcessor class used to turn wav files into spectrograms and mfccs

# Author E. Omer Gul

from cmath import log
from dbm.ndbm import library
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


class AudioPreprocessor():
    
    @staticmethod
    def wavToSpectrogram(file_path, n_fft=2048, hop_length=512, plot=False):

        signal, sr = librosa.load(file_path, sr=22050)
        stft = librosa.core.stft(signal,hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)

        librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
        if plot:
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.show()

        return log_spectrogram
    
    @staticmethod
    def wavToMfcc(file_path, n_fft=2048, hop_length=512, plot=False):

        signal, sr = librosa.load(file_path, sr=None)
        MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
        librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)

        if plot:
            plt.xlabel("Time")
            plt.ylabel("MFCC")
            plt.colorbar()
            plt.show()

        return MFCCs

    @staticmethod
    def SpectrogramToWav():
        raise NotImplementedError

