import os, sys
import numpy as np
from PIL import Image
import glob
import shutil

print("asf")
filelist = glob.glob("/home/omer/Documents/Vinci/vinci_core/local_assets/Covers/all/64x64/*")
print(filelist)
for i in filelist:
    print(i)
    songname = i.split('/')[8]
    shutil.copy(i,f"/home/omer/Documents/Vinci/vinci_core/local_assets/Covers/all/64x64/{songname} - cover (64, 64).png")

    # 