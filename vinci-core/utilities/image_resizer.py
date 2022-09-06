import os, sys
from PIL import Image

infile_path = "/home/omer/Documents/Vinci/vinci-core/local_assets/Tracks_and_Covers/bad guy/cover.png"

size = 64,64

outfile_path = os.path.splitext(infile_path)[0] + f" {size}.png"

im = Image.open(infile_path)
im.thumbnail(size, Image.ANTIALIAS)
im.save(outfile_path, "JPEG")