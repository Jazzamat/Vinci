# main program for vinci core. 

# Author E. Omer Gul

from audio_preprocessor import AudioPreprocessor

file_path = "/home/omer/Documents/Vinci/vinci-core/local_assets/Tracks_and_Covers/bad guy/bad guy [ZD6rXLXZOEI].wav"


AudioPreprocessor.wavToSpectrogram(file_path)
AudioPreprocessor.wavToMfcc(file_path)