import os, sys
import numpy as np
from PIL import Image
import glob

class ImageLoader:

    @staticmethod
    def load_covers():

    
        filelist = glob.glob("/home/omer/Documents/Vinci/vinci-core/local_assets/Tracks_and_Covers/*/cover.png")
        images_array = np.array([np.array(Image.open(fname)) for fname in filelist])
        images_array = images_array.astype("float32")/255





        return images_array