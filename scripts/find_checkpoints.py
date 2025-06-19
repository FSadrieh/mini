import os
from glob import glob
import shutil

LOG_DIR = "/data/shared/ld/mini/torchtune/llama3_2_1B"
DELETE_EMPTY_DIRS = False


def main():
    # Find all checkpoint directories
    checkpoint_dirs = glob(os.path.join(LOG_DIR, "*"))

    # Filter out empty directories
    for d in checkpoint_dirs:
        ls = os.listdir(d)
        if "epoch_0" not in ls:
            print(f"Found empty directory: {d}, with {ls}")
            if DELETE_EMPTY_DIRS:
                print(f"Deleting empty directory: {d}")
                shutil.rmtree(d)
        else:
            print(f"Found checkpoint directory: {d}, with {ls}")


if __name__ == "__main__":
    main()
