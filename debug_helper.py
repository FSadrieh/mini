from datasets import load_dataset
from datasets import Dataset

processed_train_path = "data/processed_meta-llama_Llama-3.2-1B_0_8/train"
processed_val_path = "data/processed_meta-llama_Llama-3.2-1B_0_64/validation"
tokenized_train_path = "data/tokenized_meta-llama_Llama-3.2-1B_0/train"
tokenized_val_path = "data/tokenized_meta-llama_Llama-3.2-1B_0/validation"

# CHANGE DATASET PATH HERE
dataset_path = tokenized_train_path

# CHANGE HERE TO UP/DOWNLOAD DATASET
download = True

if download:
    # Download the dataset
    dataset = load_dataset(f"frederic-sadrieh/mini-{dataset_path.replace('/', '-')}")["train"]
    # Save the dataset to disk
    dataset.save_to_disk(dataset_path)

else:
    # Load the dataset from disk
    dataset = Dataset.load_from_disk(dataset_path)
    dataset.push_to_hub(f"frederic-sadrieh/mini-{dataset_path.replace('/', '-')}")