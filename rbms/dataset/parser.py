import argparse


def add_args_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Args:
    parser: argparse.ArgumentParser:
    """
    dataset_args = parser.add_argument_group("Dataset")
    dataset_args.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to a data file (type should be .h5 or .fasta)",
    )
    dataset_args.add_argument(
        "--test_dataset",
        type=str,
        required=False,
        default=None,
        help="Path to test dataset file (type should be .h5 or .fasta)",
    )
    dataset_args.add_argument(
        "--subset_labels",
        nargs="*",
        default=None,
        type=int,
        help="(Defaults to None). The subset of labels to use during training. None means all dataset.",
    )
    dataset_args.add_argument(
        "--train_size",
        type=float,
        default=0.6,
        help="(Defaults to 0.6). The proportion of the dataset to use as training set.",
    )
    dataset_args.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="(Defaults to None). The proportion of the dataset to use as testing set.",
    )
    dataset_args.add_argument(
        "--use_weights",
        default=False,
        action="store_true",
        help="Compute the weights associated to each sequence.",
    )
    dataset_args.add_argument(
        "--alphabet",
        type=str,
        default="protein",
        choices=["protein", "rna", "dna"],
        help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.",
    )
    dataset_args.add_argument(
        "--binarize",
        default=False,
        action="store_true",
        help="(Defaults to False). Binarize the dataset.",
    )
    dataset_args.add_argument(
        "--seed",
        default=None,
        type=int,
        help="(Defaults to None). Seed to split the dataset.",
    )
    return parser
