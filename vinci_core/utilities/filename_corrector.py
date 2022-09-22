import os, sys
import numpy as np
from PIL import Image
import glob


print("asf")
filelist = glob.glob("/home/omer/Documents/Vinci/vinci_core/local_assets/Tracks_and_Covers/*")
print(filelist)
for i in filelist:
    print(i)
    os.rename(i,i.replace('-','_').replace('|','_'))


# images_array = np.array([np.array(Image.open(fname)) for fname in filelist])






