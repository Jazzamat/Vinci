import os, sys
from PIL import Image
import glob



filelist = glob.glob("/home/omer/Vinci/vinci-core/utilities/local_assets/Tracks_and_Covers/*/cover.png")

size = 64,64

for fname in filelist:

    outfile_path = os.path.splitext(fname)[0] + f" {size}.png"

    im = Image.open(fname)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(outfile_path, "JPEG")