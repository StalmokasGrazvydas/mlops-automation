import os, argparse, math, random, shutil
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True, help="Three input folders")
    parser.add_argument("--training_data_output", required=True)
    parser.add_argument("--testing_data_output", required=True)
    parser.add_argument("--split_size", type=int, default=20, help="Test percent (e.g., 20)")
    args = parser.parse_args()

    random.seed(42)
    os.makedirs(args.training_data_output, exist_ok=True)
    os.makedirs(args.testing_data_output, exist_ok=True)

    for ds in args.datasets:
        files = []
        for pat in ("*.jpg","*.jpeg","*.png"):
            files += glob(os.path.join(ds, pat))
        random.shuffle(files)
        n_test = math.ceil(len(files) * (args.split_size/100.0))
        test_files = files[:n_test]
        train_files = files[n_test:]

        for src in test_files:
            shutil.copy2(src, os.path.join(args.testing_data_output, os.path.basename(src)))
        for src in train_files:
            shutil.copy2(src, os.path.join(args.training_data_output, os.path.basename(src)))

if __name__ == "__main__":
    main()
