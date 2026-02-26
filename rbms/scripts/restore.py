# import argparse

# import h5py
# import torch

# from rbms import get_saved_updates
# from rbms.dataset import load_dataset
# from rbms.map_model import map_model
# from rbms.optim import setup_optim
# from rbms.parser import (
#     add_args_pytorch,
#     add_args_saves,
#     add_args_train,
#     add_grad_args,
#     add_sampling_args,
#     match_args_dtype,
#     remove_argument,
# )
# from rbms.training.pcd import train
# from rbms.training.utils import get_checkpoints, restore_training


# def create_parser_restore():
#     parser = argparse.ArgumentParser(
#         description="Restore the training of a Restricted Boltzmann Machine"
#     )
#     dataset_args = parser.add_argument_group("Dataset")
#     dataset_args.add_argument(
#         "-d",
#         "--dataset",
#         type=str,
#         required=True,
#         help="Path to a data file (type should be .h5 or .fasta)",
#     )
#     dataset_args.add_argument(
#         "--test_dataset",
#         type=str,
#         required=False,
#         default=None,
#         help="Path to test dataset file (type should be .h5 or .fasta)",
#     )
#     parser = add_args_train(parser)
#     parser = add_grad_args(parser)
#     parser.add_argument(
#         "--update",
#         default=None,
#         type=int,
#         help="(Defaults to None). Which update to restore from. If None, the last update is used.",
#     )
#     remove_argument(parser, "no_center")
#     remove_argument(parser, "normalize_grad")

#     parser = add_sampling_args(parser)
#     parser = add_args_saves(parser)
#     parser = add_args_pytorch(parser)
#     remove_argument(parser, "use_torch")
#     return parser


# def recover_args(
#     args: dict,
# ) -> tuple[
#     dict[str, str],
#     dict[str, int | float],
#     dict[str, bool | float],
#     dict[str, int | float],
#     dict[str, str | torch.dtype],
# ]:
#     with h5py.File(args["filename"], "r") as f:
#         # dataset
#         args_dataset = {
#             "dataset_name": args["dataset"],
#             "test_dataset_name": args["test_dataset"],
#         }
#         dataset = f["dataset_args"]
#         if "subset_labels" in dataset.keys():
#             args_dataset["subset_labels"] = dataset["subset_labels"][()]
#         else:
#             args_dataset["subset_labels"] = None
#         args_dataset["train_size"] = dataset["train_size"][()].item()
#         args_dataset["test_size"] = dataset["test_size"][()].item()

#         args_dataset["use_weights"] = dataset["use_weights"][()].item()
#         args_dataset["alphabet"] = dataset["alphabet"][()].decode()
#         args_dataset["remove_duplicates"] = dataset["remove_duplicates"][()].item()
#         args_dataset["seed"] = dataset["seed"][()].item()

#         # grad
#         args_grad = {}
#         grad = f["grad_args"]
#         ## Default args
#         args_grad["no_center"] = grad["no_center"][()].item()
#         args_grad["normalize_grad"] = grad["normalize_grad"][()].item()
#         ## Can be overriden
#         args_grad["max_norm_grad"] = args["max_norm_grad"]
#         if args_grad["max_norm_grad"] is None:
#             args_grad["max_norm_grad"] = grad["max_norm_grad"][()].item()
#         args_grad["L1"] = args["L1"]
#         if args_grad["L1"] is None:
#             args_grad["L1"] = grad["L1"][()].item()
#         args_grad["L2"] = args["L2"]
#         if args_grad["L2"] is None:
#             args_grad["L2"] = grad["L2"][()].item()

#         # sampling
#         args_sampling = {}
#         sampling = f["sampling_args"]
#         args_sampling["gibbs_steps"] = args["gibbs_steps"]
#         if args_sampling["gibbs_steps"] is None:
#             args_sampling["gibbs_steps"] = sampling["gibbs_steps"][()].item()
#         args_sampling["beta"] = args["beta"]
#         if args_sampling["beta"] is None:
#             args_sampling["beta"] = sampling["beta"][()].item()

#         # train
#         args_train = {}
#         train_args = f["train_args"]
#         args_train["optim"] = args["optim"]
#         args_train["num_updates"] = args["num_updates"]
#         if args_train["optim"] is None:
#             args_train["optim"] = train_args["optim"][()].decode()
#         args_train["learning_rate"] = args["learning_rate"]
#         if args_train["learning_rate"] is None:
#             args_train["learning_rate"] = train_args["learning_rate"][()]
#         args_train["batch_size"] = args["batch_size"]
#         if args_train["batch_size"] is None:
#             args_train["batch_size"] = train_args["batch_size"][()].item()
#         args_train["update"] = args["update"]
#         if args_train["update"] is None:
#             args_train["update"] = get_saved_updates(args["filename"])[-1]
#         args_train["mult_optim"] = args["mult_optim"]
#         args_train["training_type"] = args["training_type"]
#         if args_train["training_type"] is None:
#             args_train["training_type"] = train_args["training_type"][()].decode()
#         if args_train["max_lr"] is None:
#             args_train["max_lr"] = train_args["max_lr"][()].item()
#         args_train["scale_lr"] = args["scale_lr"]

#         # Torch
#         args_torch = {}
#         args_torch["device"] = args["device"]
#         args_torch["dtype"] = args["dtype"]

#         # save
#         args_save = {}
#         args_save["filename"] = args["filename"]
#         save = f["save_args"]
#         args_save["n_save"] = args["n_save"]
#         if args_save["n_save"] is None:
#             args_save["n_save"] = save["n_save"][()].item()
#         args_save["spacing"] = args["spacing"]
#         if args_save["spacing"] is None:
#             args_save["spacing"] = save["spacing"][()]
#         return (args_dataset, args_save, args_train, args_grad, args_sampling, args_torch)


# def main():
#     torch.set_float32_matmul_precision("high")
#     torch.backends.cudnn.benchmark = True
#     parser = create_parser_restore()
#     args = parser.parse_args()
#     args = vars(args)
#     args = match_args_dtype(args)
#     args_dataset, args_save, args_train, args_grad, args_sampling, args_torch = (
#         recover_args(args)
#     )
#     checkpoints = get_checkpoints(
#         num_updates=args_train["num_updates"],
#         n_save=args_save["n_save"],
#         spacing=args_save["spacing"],
#     )
#     train_dataset, test_dataset = load_dataset(
#         dataset_name=args_dataset["dataset_name"],
#         test_dataset_name=args_dataset["test_dataset_name"],
#         subset_labels=args_dataset["subset_labels"],
#         use_weights=args_dataset["use_weights"],
#         alphabet=args_dataset["alphabet"],
#         remove_duplicates=args_dataset["remove_duplicates"],
#         **args_torch,
#     )
#     (
#         params,
#         parallel_chains,
#         target_update,
#         elapsed_time,
#         train_dataset,
#         test_dataset,
#     ) = restore_training(
#         train_dataset=train_dataset,
#         test_dataset=test_dataset,
#         args_save=args_save,
#         args_train=args_train,
#         args_dataset=args_dataset,
#         args_torch=args_torch,
#         map_model=map_model,
#     )
#     optimizer = setup_optim(args_train["optim"], args_train, params)
#     train(
#         train_dataset=train_dataset,
#         test_dataset=test_dataset,
#         params=params,
#         parallel_chains=parallel_chains,
#         optimizer=optimizer,
#         curr_update=target_update,
#         elapsed_time=elapsed_time,
#         checkpoints=checkpoints,
#         args_save=args_save,
#         args_train=args_train,
#         args_grad=args_grad,
#         args_sampling=args_sampling,
#     )


# if __name__ == "__main__":
#     main()
