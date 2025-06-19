import os
from glob import glob
import shutil

LOG_DIR = "/data/shared/ld/mini/torchtune/llama3_2_1B"
DELETE_EMPTY_DIRS = True

def main():
    # Find all checkpoint directories
    checkpoint_dirs = glob(os.path.join(LOG_DIR, "*"))


    # Filter out empty directories
    for d in checkpoint_dirs:
        ls = os.listdir(d)
        if "epoch_0" not in ls and DELETE_EMPTY_DIRS:
            print(f"Deleting empty directory: {d}")
            shutil.rmtree(d)
        else:
            print(f"Found checkpoint directory: {d}, with {ls}")



if __name__ == "__main__":
    main()