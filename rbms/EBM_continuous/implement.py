import torch
from torch import Tensor


def _compute_energy_visibles(
    energy: torch.nn.Module,
    v: Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    v = v.to(device=device, dtype=dtype)
    return energy(v).view(-1)


def _init_chains(
    num_samples: int,
    num_visibles: int,
    device: torch.device | str,
    dtype: torch.dtype,
    data_mean: Tensor,
    data_std: Tensor,
    start_v: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        visible = data_mean.view(1, -1) + data_std.view(1, -1) * torch.randn(
            size=(num_samples, num_visibles),
            device=device,
            dtype=dtype,
        )
    else:
        visible = start_v.to(device=device, dtype=dtype)

    return visible, visible


def _compute_gradient(
    energy: torch.nn.Module,
    v_data: Tensor,
    w_data: Tensor,
    v_chain: Tensor,
    w_chain: Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    centered: bool = True,
) -> None:
    v_data = v_data.to(device=device, dtype=dtype)
    v_chain = v_chain.to(device=device, dtype=dtype)
    w_data = w_data.to(device=device, dtype=dtype).view(-1)
    w_chain = w_chain.to(device=device, dtype=dtype).view(-1)

    data_weights = w_data / w_data.sum()
    chain_weights = w_chain / w_chain.sum()

    data_energy = energy(v_data).view(-1)
    chain_energy = energy(v_chain).view(-1)

    objective = -(data_energy * data_weights).sum() + (
        chain_energy * chain_weights
    ).sum()

    energy.zero_grad(set_to_none=True)
    objective.backward()


def _energy_and_grad(
    energy: torch.nn.Module,
    visible: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    visible_grad_input = visible.detach().requires_grad_(True)
    energy_value = beta * energy(visible_grad_input).view(-1)
    grad = torch.autograd.grad(energy_value.sum(), visible_grad_input)[0]
    return energy_value.detach(), grad.detach()


def _sample_state_hmc(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    step_size: float = 1e-2,
    num_leapfrog_steps: int = 10,
    mass: float = 1.0,
    clamp: tuple[float, float] | None = None,
) -> dict[str, Tensor]:
    visible = chains["visible"].clone()
    weights = chains["weights"].clone()

    if step_size <= 0:
        raise ValueError(f"HMC step_size must be positive, got {step_size}.")
    if num_leapfrog_steps <= 0:
        raise ValueError(
            f"HMC num_leapfrog_steps must be positive, got {num_leapfrog_steps}."
        )
    if mass <= 0:
        raise ValueError(f"HMC mass must be positive, got {mass}.")

    acceptances = []
    momentum_std = mass ** 0.5

    for _ in range(n_steps):
        start_visible = visible.detach()
        start_momentum = momentum_std * torch.randn_like(start_visible)

        current_energy, grad = _energy_and_grad(energy, start_visible, beta=beta)
        current_kinetic = 0.5 * start_momentum.square().sum(dim=1) / mass

        proposal_visible = start_visible
        proposal_momentum = start_momentum - 0.5 * step_size * grad

        for leapfrog_step in range(num_leapfrog_steps):
            proposal_visible = proposal_visible + step_size * proposal_momentum / mass

            if clamp is not None:
                lo, hi = clamp
                proposal_visible = proposal_visible.clamp(lo, hi)

            proposed_energy, grad = _energy_and_grad(
                energy,
                proposal_visible,
                beta=beta,
            )

            if leapfrog_step != num_leapfrog_steps - 1:
                proposal_momentum = proposal_momentum - step_size * grad

        proposal_momentum = proposal_momentum - 0.5 * step_size * grad
        proposal_momentum = -proposal_momentum

        proposed_kinetic = 0.5 * proposal_momentum.square().sum(dim=1) / mass
        log_acceptance = (
            -proposed_energy
            - proposed_kinetic
            + current_energy
            + current_kinetic
        )

        with torch.no_grad():
            accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
            visible = torch.where(accept[:, None], proposal_visible.detach(), start_visible)
            acceptances.append(accept.float().mean())

    return {
        "visible": visible.detach(),
        "visible_mag": visible.detach(),
        "weights": weights,
        "acceptance": torch.stack(acceptances).mean() if acceptances else torch.nan,
    }
