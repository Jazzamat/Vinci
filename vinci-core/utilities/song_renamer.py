import os, sys


import glob


filelist = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/*.wav")
print(filelist)
for filepath in filelist:
    
    # Extract the directory path from the full file path
    directory = os.path.dirname(filepath)

    # Construct the new file path with the same directory but with 'song.wav' as the file name
    new_filepath = os.path.join(directory, 'song.wav')

    # Rename the file
    os.rename(filepath, new_filepath)

# images_array = np.array([np.array(Image.open(fname)) for fname in filelist])






