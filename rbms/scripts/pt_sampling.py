import argparse

import h5py
import torch

from rbms.parser import add_args_pytorch
from rbms.sampling.pt import pt_sampling
from rbms.utils import check_file_existence, get_saved_updates, load_params


def create_parser():
    parser = argparse.ArgumentParser(
        "Parallel Tempering sampling on the provided model"
    )
    parser.add_argument("-i", "--filename", type=str, help="Model to use for sampling")
    parser.add_argument(
        "-o", "--out_file", type=str, help="Path to save the samples after generation"
    )
    parser.add_argument(
        "--num_samples",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of generated samples.",
    )
    parser.add_argument(
        "--target_acc_rate",
        default=0.3,
        type=float,
        help="(Defaults to 0.3). Target acceptance rate between two consecutive models.",
    )
    parser.add_argument(
        "--it_mcmc",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of MCMC steps to perform.",
    )
    parser.add_argument(
        "--index",
        default=False,
        action="store_true",
        help="(Defaults to False). Save the starting index of the chains during sampling. Useful to compute mixing time.",
    )
    parser.add_argument(
        "--increment",
        default=1,
        type=int,
        help="(Defaults to 1). Number of Gibbs steps to perform between each swap.",
    )
    parser = add_args_pytorch(parser)

    return parser


def run_pt(
    filename: str,
    out_file: str,
    num_samples: int,
    it_mcmc: int,
    target_acc_rate: float,
    increment: int,
    save_index: bool,
    device,
    dtype,
):
    check_file_existence(out_file)

    age = get_saved_updates(filename)[-1]
    params = load_params(filename=filename, index=age, device=device, dtype=dtype)

    list_chains, inverse_temperatures, index = pt_sampling(
        it_mcmc=it_mcmc,
        increment=increment,
        target_acc_rate=target_acc_rate,
        num_chains=num_samples,
        params=params,
        out_file=out_file,
        save_index=save_index,
    )

    for i in range(len(list_chains)):
        with h5py.File(out_file, "a") as f:
            f[f"gen_{i}"] = list_chains[i].visible.cpu().numpy()
    with h5py.File(out_file, "a") as f:
        f["sel_beta"] = inverse_temperatures


def main():
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    match args["dtype"]:
        case "int":
            args["dtype"] = torch.int64
        case "half":
            args["dtype"] = torch.float16
        case "float":
            args["dtype"] = torch.float32
        case "double":
            args["dtype"] = torch.float64
    run_pt(
        filename=args["filename"],
        out_file=args["out_file"],
        num_samples=args["num_samples"],
        it_mcmc=args["it_mcmc"],
        target_acc_rate=args["target_acc_rate"],
        increment=args["increment"],
        save_index=args["index"],
        device=args["device"],
        dtype=args["dtype"],
    )


if __name__ == "__main__":
    main()
