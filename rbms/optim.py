import numpy as np
import torch

# from ptt.optim.cossim import SGD_cossim
from torch import Tensor
from torch.optim import SGD, Adam, Optimizer

from rbms.classes import EBM, Sampler


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
            # learning_rate *= 1 + cosine_similarity.item() * 0.001
            group["lr"] = min(self.max_lr, learning_rate)
            self.prev_grad = curr_grad.clone()
        return super().step(closure)


class PTTSpeedController:
    """Learning rate control on the displacement of the model "for PTT".

    The swap acceptance between two models p_theta and p_theta' is ~erfc(sigma / 2),
    with sigma the std over the samples of F_theta'(x) - F_theta(x) (F: free energy of
    the visibles). Every `window` updates, the controller measures the displacement of
    the model since the previous measurement on the current chains of the sampler,
        v = std_x[F_now(x) - F_before(x)] / window,
    and multiplies the learning rate of all the optimizers by (target / v)^gamma
    (clipped to [1 / max_factor, max_factor]), so that the model moves by ~target per
    update. The displacement is the real one (after the projections of the parameters),
    over several updates: the noise of the gradient, which cancels between updates,
    does not count (unlike the norm of the step of a single update).

    With several optimizers (one per parameter, --mult_optim), the learning rates are
    also distributed between the parameters: at each update, the lr of parameter i is
    multiplied by exp(relative_rate * (cos_i - mean_j cos_j)), with cos_i the cosine
    between its gradient and the previous one (coherent steps: the lr can grow;
    oscillating steps: it must decrease). The factors are centered on their mean, so
    they only change the ratios between the learning rates, and the scale stays set by
    the displacement of the model.
    """

    def __init__(
        self,
        params: EBM,
        sampler: Sampler,
        target: float = 0.01,
        window: int = 10,
        gamma: float = 0.5,
        max_factor: float = 2.0,
        relative_rate: float = 0.01,
    ):
        self.params = params
        self.sampler = sampler
        self.target = target
        self.window = window
        self.gamma = gamma
        self.max_factor = max_factor
        self.relative_rate = relative_rate
        self.optimizers: list[Optimizer] = []
        self.prev_grads: list[Tensor] | None = None
        self.last_cosines: np.ndarray | None = None
        self.num_steps = 0
        self.snapshot: EBM | None = None
        self.last_speed: float | None = None

    @torch.no_grad()
    def _distribute(self):
        """Relative learning rates of the parameters from the cosine between their
        current and previous gradients (the gradients of all the parameters are already
        computed when the first optimizer steps)."""
        grads = [
            torch.cat([p.grad.flatten() for group in opt.param_groups for p in group["params"]])
            for opt in self.optimizers
        ]
        if self.prev_grads is not None:
            cosines = np.array(
                [
                    (g @ g0 / (g.norm() * g0.norm() + 1e-30)).item()
                    for g, g0 in zip(grads, self.prev_grads)
                ]
            )
            self.last_cosines = cosines
            factors = np.exp(self.relative_rate * (cosines - cosines.mean()))
            for opt, factor in zip(self.optimizers, factors):
                for group in opt.param_groups:
                    group["lr"] = min(group["lr"] * factor, getattr(opt, "max_lr", np.inf))
        self.prev_grads = [g.clone() for g in grads]

    @torch.no_grad()
    def step(self):
        """Called once per update, after the gradients are computed and before the
        parameters are changed."""
        if len(self.optimizers) > 1 and self.relative_rate > 0:
            self._distribute()
        if self.snapshot is None:
            self.snapshot = self.params.clone()
            self.num_steps = 0
            return
        self.num_steps += 1
        if self.num_steps < self.window:
            return
        chains = self.sampler.get_curr_conf()
        delta = self.params.compute_energy_visibles(
            chains
        ) - self.snapshot.compute_energy_visibles(chains)
        speed = delta.std().item() / self.num_steps
        self.last_speed = speed
        if speed > 0 and np.isfinite(speed):
            factor = (self.target / speed) ** self.gamma
        else:
            factor = self.max_factor
        factor = min(max(factor, 1 / self.max_factor), self.max_factor)
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = min(group["lr"] * factor, getattr(opt, "max_lr", np.inf))
        self.snapshot = self.params.clone()
        self.num_steps = 0


class SGD_cossim_ptt(SGD):
    """SGD whose learning rate is set by a shared `PTTSpeedController`. With several
    optimizers (one per parameter), they share the controller, which sets the scale of
    all the learning rates and their ratios; the first one calls it once per update."""

    def __init__(self, params, lr=0.001, maximize=True, **kwargs):
        super().__init__(params, lr, maximize=maximize, **kwargs)
        self.controller: PTTSpeedController | None = None
        self.max_lr = np.inf

    def step(self, closure=None):
        if self.controller is not None and self.controller.optimizers[0] is self:
            self.controller.step()
        return super().step(closure)


@torch.no_grad()
def initial_growth_rate(params: EBM, data: Tensor, num_samples: int = 20000) -> float:
    """Growth rate of the weights at the beginning of the training.

    Near W = 0 (binary {0, 1} hiddens with zero bias, p(h = 1 | v) ~ 1/2 + v.W / 4) the
    gradient of W is ~ (C_data - C_model) W / 4, with C the covariance of the visibles,
    so W grows by a factor ~exp(lr * mu) per update, with mu the largest eigenvalue of
    (C_data - C_model) / 4. C_model is that of the independent model of `params`.
    """
    perm = torch.randperm(data.shape[0], device=data.device)[:num_samples]
    v_data = data[perm].reshape(len(perm), -1).to(torch.float64)
    v_model = params.independent_model().init_chains(num_samples)["visible"]
    v_model = v_model.reshape(num_samples, -1).to(device=v_data.device, dtype=torch.float64)
    diff = (torch.cov(v_data.T) - torch.cov(v_model.T)) / 4
    return torch.linalg.eigvalsh(diff).max().item()


def setup_optim(optim: str, args: dict, params: EBM, sampler: Sampler) -> list[Optimizer]:
    match optim:
        case "sgd":
            optim_class = SGD
        case "cossim":
            optim_class = SGD_cossim
        case "cossim_ptt":
            optim_class = SGD_cossim_ptt
        case "adam":
            optim_class = Adam
        case "ngd":
            optim_class = lambda p, **kwargs: NGD(
                p,
                update_freq=1,
                warm_start=True,
                update_biases=params.train_biases,
                max_dF=args.get("ptt_speed") or 0.01,
                **kwargs,
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
        if isinstance(opt, (SGD_cossim, SGD_cossim_ptt, NGD)):
            opt.max_lr = max_lr
    if args["optim"] == "cossim_ptt":
        controller = PTTSpeedController(
            params=params,
            sampler=sampler,
            target=args.get("ptt_speed") or 0.01,
            window=args.get("ptt_window") or 10,
            relative_rate=args.get("ptt_relative")
            if args.get("ptt_relative") is not None
            else 0.01,
        )
        controller.optimizers = optimizer
        for opt in optimizer:
            opt.controller = controller

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
    if args["optim"] == "ngd":
        for opt in optimizer:
            assert isinstance(opt, NGD)
            opt.set_sampler(sampler)
            opt.set_model(params)
    return optimizer


class NGD(Optimizer):
    # Added 'update_biases=True' flag to the initialization
    def __init__(
        self,
        params,
        lr=0.001,
        cg_steps=20,
        init_reg=1,
        update_freq=1,
        warm_start=False,
        maximize=True,
        update_biases=True,
        max_lr=0.01,
        max_dF=None,
    ):
        self._model = None
        self._sampler = None
        defaults = dict(
            lr=lr,
            cg_steps=cg_steps,
            reg=init_reg,
            update_freq=update_freq,
            warm_start=warm_start,
            maximize=maximize,
            update_biases=update_biases,
            step=0,
            max_lr=max_lr,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["last_dt"] = torch.zeros_like(p.data)
        self.max_lr = max_lr
        self.rescale_eigenvalue_lr = True
        # Limit of std_x[F_new(x) - F(x)] per update on the chains of the sampler (None: no
        # limit). The natural gradient can be huge where the Fisher matrix vanishes (Gaussian
        # hiddens at W = 0), and the swap acceptance of PTT is ~erfc(std / 2)
        self.max_dF = max_dF
        self.last_lr = None

    @torch.no_grad()
    def _limit_scale(self, direction, params, steps, max_iter=20):
        """Largest factor s <= 1 such that the step sum_i s * lr_i * delta_i, with steps the
        list of (index of the parameter, delta_i, lr_i), changes the free energy of the chains
        of the sampler by at most max_dF (std over the chains), on the exact free energy. The
        change of F is not linear in the step (quadratic for large steps with Gaussian
        hiddens): the factor is first reduced until the limit holds, then refined by a
        bisection in log scale."""
        chains = self._sampler.get_curr_conf()
        energy = self._model.compute_energy_visibles(chains)

        def std_dF(factor):
            model = self._model.clone()
            new_params = model.parameters()
            for i, dt, lr in steps:
                new_params[i].add_(dt, alpha=direction * factor * lr)
            return (model.compute_energy_visibles(chains) - energy).std().item()

        def valid(factor):
            std = std_dF(factor)
            return np.isfinite(std) and std <= self.max_dF, std

        ok, std = valid(1.0)
        if ok:
            return 1.0
        # Reduce until the limit holds (linear estimate: too small if F grows faster)
        hi, lo = 1.0, 1.0
        for _ in range(max_iter):
            hi = lo
            lo *= 0.9 * self.max_dF / std if np.isfinite(std) and std > 0 else 1e-3
            ok, std = valid(lo)
            if ok:
                break
        if not ok:
            return 0.0
        # Bisection in log scale between lo (valid) and hi (not valid), up to a factor 1.05
        for _ in range(max_iter):
            if hi / lo < 1.05:
                break
            mid = np.sqrt(lo * hi)
            if valid(mid)[0]:
                lo = mid
            else:
                hi = mid
        return lo

    def set_sampler(self, sampler: Sampler):
        self._sampler = sampler

    def set_model(self, model: EBM):
        self._model = model

    @staticmethod
    def _score_dot(scores, p_list) -> Tensor:
        """O_k . p for each sample k, with O_k the scores (see `RBM.compute_scores`)."""
        out = 0
        for score, p in zip(scores, p_list):
            if isinstance(score, tuple):
                left, right = score
                num_samples = left.shape[0]
                out = out + (
                    (left.reshape(num_samples, -1) @ p.reshape(-1, p.shape[-1])) * right
                ).sum(1)
            else:
                out = out + score.reshape(score.shape[0], -1) @ p.reshape(-1)
        return out

    @torch.no_grad()
    def _get_adaptive_reg(self, scores, base_reg, num_elements):
        """base_reg * tr(F_1) / (2 D), with F_1 the block of the Fisher matrix of the first
        parameter of `scores` (the weight matrix of the block) and D = num_elements its
        number of elements."""
        score = scores[0]
        if isinstance(score, tuple):
            left, right = score
            num_samples = left.shape[0]
            left = left.reshape(num_samples, -1)
            mean_norm_sq = ((left**2).sum(1) * (right**2).sum(1)).mean()
            norm_mean_sq = ((left.T @ right / num_samples) ** 2).sum()
        else:
            flat = score.reshape(score.shape[0], -1)
            mean_norm_sq = (flat**2).sum(1).mean()
            norm_mean_sq = (flat.mean(0) ** 2).sum()
        trace_F = mean_norm_sq - norm_mean_sq
        return base_reg * trace_F.item() / (2 * num_elements)

    @torch.no_grad()
    def _fvp(self, p_list, scores, reg, masks=None):
        """(F + reg) p, with F the covariance of the scores over the chains. With masks (None
        or a mask per parameter), P (F + reg) P p, with P the projection on the trained
        entries (sparse training)."""
        o = self._score_dot(scores, p_list)
        o = o - o.mean()
        num_samples = o.shape[0]
        if masks is None:
            masks = [None] * len(p_list)
        out = []
        for score, p, mask in zip(scores, p_list, masks):
            if isinstance(score, tuple):
                left, right = score
                F_p = left.reshape(num_samples, -1).T @ (right * o.unsqueeze(1)) / num_samples
            else:
                F_p = score.reshape(num_samples, -1).T @ o / num_samples
            F_p = F_p.view_as(p) + reg * p
            out.append(F_p if mask is None else F_p * mask)
        return out

    @torch.no_grad()
    def _solve(self, block_params, block_grads, block_scores, reg, group, block_masks):
        """Solve (F + reg) delta = g on one block by conjugate gradient (warm start from the
        solution of the previous update). With a mask (sparse training), the system is
        restricted to the trained entries: all the iterates stay zero out of the mask."""
        if group["warm_start"] and group["step"] > 1:
            delta_theta = [
                self.state[p]["last_dt"].clone() if m is None else self.state[p]["last_dt"] * m
                for p, m in zip(block_params, block_masks)
            ]
            S_dt = self._fvp(delta_theta, block_scores, reg, block_masks)
            residual = [g_i - s_i for g_i, s_i in zip(block_grads, S_dt)]
        else:
            delta_theta = [torch.zeros_like(p) for p in block_params]
            residual = [g_i.clone() for g_i in block_grads]
        residual = [r if m is None else r * m for r, m in zip(residual, block_masks)]

        p_vec = [r_i.clone() for r_i in residual]
        r_dot_r = sum(torch.sum(r * r) for r in residual)
        self.cg_step = 0
        for _ in range(group["cg_steps"]):
            S_p = self._fvp(p_vec, block_scores, reg, block_masks)
            p_Sp = sum(torch.sum(pv * spv) for pv, spv in zip(p_vec, S_p))
            p_p = sum(torch.sum(pv * pv) for pv in p_vec)

            # Relative to |p|^2: the scale of the Fisher matrix depends on the model
            # (with Gaussian hiddens of variance 1/Nv it is tiny at the beginning)
            if not p_Sp.item() > 1e-12 * p_p.item():
                print("CG broke due to non-positive curvature.")
                break

            alpha = (r_dot_r / p_Sp).item()
            for dt, pv in zip(delta_theta, p_vec):
                dt.add_(pv, alpha=alpha)
            for r, spv in zip(residual, S_p):
                r.sub_(spv, alpha=alpha)

            new_r_dot_r = sum(torch.sum(r * r) for r in residual)
            if new_r_dot_r.item() < 1e-20:
                print("CG broke due to tiny residual norm.")
                break

            beta = (new_r_dot_r / r_dot_r).item()
            for pv, r in zip(p_vec, residual):
                pv.mul_(beta).add_(r)
            r_dot_r = new_r_dot_r
            self.cg_step += 1

        for p_tensor, dt in zip(block_params, delta_theta):
            self.state[p_tensor]["last_dt"].copy_(dt)
        return delta_theta

    @torch.no_grad()
    def step(self, scale=1, closure=None):
        assert self._model is not None
        assert self._sampler is not None
        v_chain = self._sampler.get_curr_conf()["visible"]
        for group in self.param_groups:
            group["step"] += 1
            params = group["params"]
            update_biases = group["update_biases"]

            g = [p.grad.clone() for p in params]
            # Sparse training: mask of the trained couplings of the weight matrix
            weight_mask = getattr(self._model, "weight_mask", None)
            masks = [
                weight_mask if p is self._model.weight_matrix else None for p in params
            ]

            # Blocks of the Fisher matrix (block diagonal), each one with its own step limit.
            # Only the weight matrix is trained when the model does not train its biases
            blocks = self._model.ngd_blocks()
            if not update_biases:
                blocks = [
                    [i for i in block if params[i] is self._model.weight_matrix]
                    for block in blocks
                ]
            blocks = [block for block in blocks if len(block) > 0]
            assert len(blocks) > 0

            if group["step"] % group["update_freq"] == 0 or group["step"] == 1:
                # Scores of all the parameters, in the order of the parameter group
                scores = self._model.compute_scores(v_chain)
                assert len(scores) == len(params)
                block_deltas = []
                self.reg = []
                for block in blocks:
                    reg = scale * self._get_adaptive_reg(
                        [scores[i] for i in block], group["reg"], params[block[0]].numel()
                    )
                    self.reg.append(reg)
                    block_deltas.append(
                        self._solve(
                            [params[i] for i in block],
                            [g[i] for i in block],
                            [scores[i] for i in block],
                            reg,
                            group,
                            [masks[i] for i in block],
                        )
                    )
            else:
                block_deltas = [[g[i] for i in block] for block in blocks]

            indices = [i for block in blocks for i in block]
            delta_theta = [dt for deltas in block_deltas for dt in deltas]
            active_grads = [g[i] for i in indices]

            # --- Raise learning rate based on cosine similarity ---
            dot_product = sum(
                torch.sum(dt * g_i) for dt, g_i in zip(delta_theta, active_grads)
            )
            norm_dt = torch.sqrt(sum(torch.sum(dt**2) for dt in delta_theta))
            norm_g = torch.sqrt(sum(torch.sum(g_i**2) for g_i in active_grads))
            self.cos_sim = (dot_product / (norm_dt * norm_g + 1e-20)).item()
            group["lr"] *= 1 + self.cos_sim * 0.001
            group["lr"] = min(group["lr"], self.max_lr)

            # Rescale eigenvalues: the step of the weight matrix along its main direction is
            # divided by Nv while the gain of the linear response of the model is below 1, i.e.
            # before the first transition (see `RBM.ngd_coupling`)
            if self.rescale_eigenvalue_lr:
                self.rescale_eigenvalue_lr = (
                    torch.linalg.matrix_norm(self._model.ngd_coupling(), ord=2) ** 2 < 1
                ).item()
            if self.rescale_eigenvalue_lr:
                k = next(k for k, i in enumerate(indices) if params[i] is self._model.weight_matrix)
                _, eig_vector = torch.lobpcg(delta_theta[k] @ delta_theta[k].T, k=1, largest=True)
                contrib = ((delta_theta[k].T @ eig_vector) @ eig_vector.T).T
                delta_theta[k] = delta_theta[k] - contrib + contrib / self._model.num_visibles
                if weight_mask is not None:
                    delta_theta[k] = delta_theta[k] * weight_mask

            # ------------------------------------------------------
            direction = 1 if group["maximize"] else -1
            # A float: group["lr"] can be a tensor, which the limit must not modify in place
            lr = float(group["lr"])
            block_of = [b for b, block in enumerate(blocks) for _ in block]
            lrs = [lr] * len(blocks)
            if self.max_dF is not None:
                # Limit of this step only (group["lr"] keeps its own evolution): each block is
                # limited on its own, then the joint step if the blocks together exceed it
                for b in range(len(blocks)):
                    steps = [
                        (i, dt, lr)
                        for i, dt, bb in zip(indices, delta_theta, block_of)
                        if bb == b
                    ]
                    lrs[b] = lr * self._limit_scale(direction, params, steps)
                if len(blocks) > 1:
                    steps = [
                        (i, dt, lrs[bb]) for i, dt, bb in zip(indices, delta_theta, block_of)
                    ]
                    factor = self._limit_scale(direction, params, steps)
                    lrs = [lr_b * factor for lr_b in lrs]
            self.last_lr = lrs if len(lrs) > 1 else lrs[0]
            for i, dt, b in zip(indices, delta_theta, block_of):
                params[i].add_(dt, alpha=direction * lrs[b])
