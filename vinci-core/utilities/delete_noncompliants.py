import os
import glob
import shutil

# Path to the directory containing the folders
base_dir = "/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers"

# Path for the backup directory
backup_dir = "/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers_Backup"

# Create the backup directory if it doesn't exist
os.makedirs(backup_dir, exist_ok=True)

# Iterate through each subfolder in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)

    # Check if the current path is a directory
    if os.path.isdir(folder_path):
        # Check for the presence of 'song.wav' and 'cover (64,64).png' in the folder
        song_exists = glob.glob(os.path.join(folder_path, 'song.wav'))
        cover_exists = glob.glob(os.path.join(folder_path, 'cover (64, 64).png'))

        # If either file is missing, move the folder to the backup directory
        if not song_exists or not cover_exists:
            backup_folder_path = os.path.join(backup_dir, folder)
            shutil.move(folder_path, backup_folder_path)
            print(f"Moved to backup: {backup_folder_path}")
