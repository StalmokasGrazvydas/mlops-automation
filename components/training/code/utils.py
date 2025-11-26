import os
from glob import glob

def list_images(folder):
    files = []
    for pat in ("*.jpg","*.jpeg","*.png"):
        files += glob(os.path.join(folder, pat))
    return sorted(files)

def label_from_name(path):
    name = os.path.basename(path).lower()
    return name.split('_')[0]  # e.g. panda_0001.jpg -> 'panda'
