import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from rbms.dataset import load_dataset
from rbms.dataset.fasta_utils import write_fasta
from rbms.dataset.utils import get_unique_indices


def create_parser():
    parser = argparse.ArgumentParser(description="Split a dataset in train/test files.")
    parser.add_argument("-i", "--filename", type=str, help="Dataset to split")
    parser.add_argument(
        "--out_train",
        type=str,
        default=None,
        help="(Defaults to None). Path to save the train dataset. If None, a generic name is used in the same folder as the input dataset.",
    )
    parser.add_argument(
        "--out_test",
        type=str,
        default=None,
        help="(Defaults to None). Path to save the test dataset. If None, a generic name is used in the same folder as the input dataset.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.6,
        help="(Defaults to 0.6). Proportion of the dataset used for the training set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(Defaults to None). The seed to split the dataset. If None, a random seed is used.",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="protein",
        help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.",
    )
    return parser


def split_data_train_test(
    input_file: str,
    output_train_file: Optional[str] = None,
    output_test_file: Optional[str] = None,
    train_size=0.6,
    seed: int = None,
    alphabet: str = "protein",
):
    dset_name = Path(input_file)
    dset_name = dset_name.resolve()

    dataset, _ = load_dataset(input_file, None, alphabet=alphabet)

    print("Removing duplicates...")
    prev_size = dataset.data.shape[0]
    unique_ind = get_unique_indices(dataset.data)
    data = dataset.data[unique_ind]
    names = dataset.names[unique_ind]
    labels = dataset.labels[unique_ind]
    
    curr_size = data.shape[0]
    print(f"    Dataset size: {prev_size} -> {curr_size} samples")
    print(f"    Removed {prev_size - curr_size} samples.")
    print("    Done")

    rng = np.random.default_rng(seed=seed)
    num_samples = data.shape[0]

    print("Shuffling and splitting dataset...")
    # Shuffle dataset
    permutation_index = rng.permutation(num_samples)
    n_sample_train = int(train_size * num_samples)

    data_train = data[permutation_index[:n_sample_train]].int().cpu().numpy()
    names_train = names[permutation_index[:n_sample_train]]
    labels_train = labels[permutation_index[:n_sample_train]].int().cpu().numpy()

    data_test = data[permutation_index[n_sample_train:]].int().cpu().numpy()
    names_test = names[permutation_index[n_sample_train:]]
    labels_test = labels[permutation_index[n_sample_train:]].int().cpu().numpy()


    print(
        f"    train_size = {data_train.shape[0]} ({100 * data_train.shape[0] / data.shape[0]}%)"
    )
    print(
        f"    test_size = {data_test.shape[0]} ({100 * data_test.shape[0] / data.shape[0]}%)"
    )
    print("    Done")

    file_format = str(dset_name).split(".")[-1]

    if output_train_file is None:
        output_train_file = (
            ".".join(str(dset_name).split(".")[:-1]) + f"_train={train_size}.{file_format}"
        )
    if output_test_file is None:
        output_test_file = (
            ".".join(str(dset_name).split(".")[:-1]) + f"_test={1 - train_size}.{file_format}"
        )

    match file_format:
        case "h5":
            print(f"Writing train dataset to '{output_train_file}'...")
            with h5py.File(output_train_file, "w") as f:
                f["samples"] = data_train
                f["labels"] = labels_train
            print("    Done")
            print(f"Writing test dataset to '{output_test_file}'...")
            with h5py.File(output_test_file, "w") as f:
                f["samples"] = data_test
                f["labels"] = labels_test
            print("    Done")            

        case "fasta":
            print(f"Writing train dataset to '{output_train_file}'...")
            write_fasta(output_train_file, names_train, data_train, numeric_input=True)
            print("    Done")
            print(f"Writing test dataset to '{output_test_file}'...")
            write_fasta(output_test_file, names_test, data_test, numeric_input=True)
            print("    Done")

        case _:
            raise ValueError(f"Wrong file format: {file_format}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    split_data_train_test(
        input_file=args["filename"],
        output_train_file=args["out_train"],
        output_test_file=args["out_test"],
        train_size=args["train_size"],
        seed=args["seed"],
        alphabet=args["alphabet"],
    )


if __name__ == "__main__":
    main()
