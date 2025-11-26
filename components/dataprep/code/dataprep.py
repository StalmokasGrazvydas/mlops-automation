import os
import argparse
from glob import glob
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output_data", type=str, help="path to output data")
    args = parser.parse_args()

    print("input data:", args.data)
    print("output folder:", args.output_data)

    os.makedirs(args.output_data, exist_ok=True)
    size = (64, 64)  # can make this dynamic later

    for file in glob(os.path.join(args.data, "*.jpg")):
        img = Image.open(file)
        img_resized = img.resize(size)
        out = os.path.join(args.output_data, os.path.basename(file))
        img_resized.save(out)

if __name__ == "__main__":
    main()
