import numpy as np
import torch

# from ptt.optim.cossim import SGD_cossim
from torch import Tensor
from torch.optim import SGD, Adam, Optimizer

from rbms.classes import EBM


class SGD_cossim(SGD):
    def __init__(
        self,
        params,
        lr=0.001,
        max_lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=True,
        foreach=None,
        differentiable=False,
        fused=None,
    ):
        super().__init__(
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        self.prev_grad = torch.concatenate([p.grad.flatten() for p in params]).flatten()
        self.max_lr = max_lr

    def step(self, closure=None):
        for group in self.param_groups:
            params = group["params"]
            learning_rate = group["lr"]
            curr_grad = torch.concatenate([p.grad.flatten() for p in params]).flatten()
            cosine_similarity = curr_grad @ self.prev_grad
            if cosine_similarity > 1e-6:
                learning_rate *= 1.002
            elif cosine_similarity < -1e-6:
                learning_rate *= 0.998
            group["lr"] = min(self.max_lr, learning_rate)
            self.prev_grad = curr_grad.clone()
        return super().step(closure)


class SR_CG(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        cg_steps=10,
        reg=1e-4,
        update_freq=1,
        warm_start=True,
        maximize=True,
    ):
        defaults = dict(
            lr=lr,
            cg_steps=cg_steps,
            reg=reg,
            update_freq=update_freq,
            warm_start=warm_start,
            maximize=maximize,
            step=0,
        )
        self._model = params
        super().__init__(params, defaults)

        # Initialize state memory for warm starts
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["last_dt"] = torch.zeros_like(p.data)

    @torch.no_grad()
    def _fvp(self, p_list, v_chain, tanh_term, model, reg, params):
        # p_dict = {}
        # for p_tensor, param_ref in zip(p_list, params):
        #     if param_ref is model.weight_matrix:
        #         p_dict["w"] = p_tensor
        #     elif param_ref is model.vbias:
        #         p_dict["v"] = p_tensor
        #     elif param_ref is model.hbias:
        #         p_dict["h"] = p_tensor

        O_dot_p = (
            ((v_chain @ model.weight_matrix) * tanh_term).sum(dim=1)
            + (v_chain @ model.vbias)
            + (tanh_term @ model.hbias)
        )

        O_dot_p_c = O_dot_p - O_dot_p.mean()

        B = v_chain.size(0)

        Sx_w = (v_chain.T @ (tanh_term * O_dot_p_c.unsqueeze(1))) / B
        Sx_v = (v_chain.T @ O_dot_p_c) / B
        Sx_h = (tanh_term.T @ O_dot_p_c) / B

        Sx_list = []
        for p_tensor, param_ref in zip(p_list, params):
            if param_ref is model.weight_matrix:
                Sx_list.append(Sx_w + reg * p_tensor)
            elif param_ref is model.vbias:
                Sx_list.append(Sx_v + reg * p_tensor)
            elif param_ref is model.hbias:
                Sx_list.append(Sx_h + reg * p_tensor)

        return Sx_list

    @torch.no_grad()
    def step(self, v_chain, closure=None):
        for group in self.param_groups:
            group["step"] += 1
            params = group["params"]
            lr = group["lr"]
            reg = group["reg"]

            g = [p.grad.clone() for p in params]

            # Lazy Preconditioning: Only run CG every 'update_freq' steps
            if group["step"] % group["update_freq"] == 0 or group["step"] == 1:
                local_field = self._model.hbias + v_chain @ self._model.weight_matrix
                tanh_term = torch.tanh(local_field)

                if group["warm_start"] and group["step"] > 1:
                    # Warm Start: Initialize with previous Natural Gradient
                    delta_theta = [self.state[p]["last_dt"].clone() for p in params]
                    S_dt = self._fvp(
                        delta_theta, v_chain, tanh_term, self._model, reg, params
                    )
                    residual = [g_i - s_i for g_i, s_i in zip(g, S_dt)]
                else:
                    # Cold Start
                    delta_theta = [torch.zeros_like(p) for p in params]
                    residual = [g_i.clone() for g_i in g]

                p_vec = [r_i.clone() for r_i in residual]
                r_dot_r = sum(torch.sum(r * r) for r in residual)

                for _ in range(group["cg_steps"]):
                    S_p = self._fvp(p_vec, v_chain, tanh_term, self._model, reg, params)
                    p_Sp = sum(torch.sum(pv * spv).item() for pv, spv in zip(p_vec, S_p))

                    if p_Sp <= 1e-8:
                        break

                    alpha = r_dot_r / p_Sp

                    for dt, pv in zip(delta_theta, p_vec):
                        dt.add_(pv, alpha=alpha)

                    for r, spv in zip(residual, S_p):
                        r.sub_(spv, alpha=alpha)

                    new_r_dot_r = sum(torch.sum(r * r).item() for r in residual)

                    if new_r_dot_r < 1e-6:
                        break

                    beta = new_r_dot_r / r_dot_r

                    for pv, r in zip(p_vec, residual):
                        pv.mul_(beta).add_(r)

                    r_dot_r = new_r_dot_r

                # Save the Natural Gradient for the next warm start
                for p_tensor, dt in zip(params, delta_theta):
                    self.state[p_tensor]["last_dt"].copy_(dt)
            else:
                # Fallback: Standard Gradient Update for intermediate steps
                delta_theta = g

            # Apply the update
            direction = 1 if group["maximize"] else -1
            for p_tensor, dt in zip(params, delta_theta):
                p_tensor.add_(dt, alpha=direction * lr)


def setup_optim(optim: str, args: dict, params: EBM) -> list[Optimizer]:
    match args["optim"]:
        case "sgd":
            optim_class = SGD
        case "cossim":
            optim_class = SGD_cossim
        case "adam":
            optim_class = Adam
        case "sr":
            optim_class = lambda p, **kwargs: SR_CG(
                p, update_freq=1, warm_start=True, **kwargs
            )
        case _:
            print(f"Unrecognized optimizer {args['optim']}, falling back to SGD.")
            optim_class = SGD
    learning_rate = args["learning_rate"]
    max_lr = args["max_lr"]
    if args["scale_lr"]:
        learning_rate /= np.sqrt(params.effective_number_variables)
        max_lr /= np.sqrt(params.effective_number_variables)

    if args["mult_optim"]:
        if not isinstance(learning_rate, Tensor):
            learning_rate = torch.tensor([learning_rate] * len(params.parameters()))
        optimizer = [
            optim_class(
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
            optim_class(
                params.parameters(),
                lr=learning_rate[0],
                maximize=True,
            )
        ]
    for opt in optimizer:
        if isinstance(opt, SGD_cossim):
            opt.max_lr = max_lr

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
