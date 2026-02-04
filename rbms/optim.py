from rbms.classes import EBM
from torch.optim import SGD, Optimizer
from torch import Tensor
import torch

def setup_optim(optim: str, args: dict, params: EBM) -> list[Optimizer]:
    match args["optim"]:
        case "sgd":
            optim = SGD
        case _:
            print(f"Unrecognized optimizer {args['optim']}, falling back to SGD.")
            optim = SGD
    learning_rate = args["learning_rate"]
    if args["mult_optim"]:
        if not isinstance(learning_rate, Tensor):
            learning_rate = torch.tensor([learning_rate] * len(params.parameters()))
        optimizer = [
            optim(
                [p],
                lr=learning_rate[i],
                maximize=True,
            )
            for i, p in enumerate(params.parameters())
        ]
    else:
        if not isinstance(learning_rate, Tensor):
            learning_rate = torch.tensor([learning_rate])
        optimizer = [
            optim(
                params.parameters(),
                lr=learning_rate[0],
                maximize=True,
            )
        ]

    if args["optim"] == "nag":
        optimizer = [
            SGD(
                opt.param_groups[0]["params"],
                lr=opt.param_groups[0]["lr"],
                maximize=True,
                momentum=0.9,
                nesterov=True,
            )
            for opt in optimizer
        ]

    return optimizer
