import os
import argparse
import subprocess
from tqdm import tqdm


def transfer(data_dir):
    for filename in tqdm(os.listdir(data_dir), total=len(os.listdir(data_dir))):
        if filename.endswith("?download=true"):
            new_filename = filename.replace("?download=true", "")
            os.rename(filename, new_filename)
        else:
            new_filename = filename

        file_path = os.path.join(data_dir, new_filename)
        if new_filename.endswith(".gz"):
            subprocess.run(["gunzip", file_path])
        elif new_filename.endswith(".zip"):
            subprocess.run(["unzip", file_path])
        elif new_filename.endswith(".tar"):
            subprocess.run(["tar", "-xvf", file_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transfer wget downloaded hf files to the right file and name format")
    parser.add_argument("--data_dir", type=str, default="./")
    args = parser.parse_args()

    transfer(args.data_dir)