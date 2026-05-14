import torch
import torch.nn.functional
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
    start_v: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        mean_visible = (
            torch.ones(
                size=(num_samples, num_visibles),
                device=device,
                dtype=dtype,
            )
            / 2
        )
        visible = torch.bernoulli(mean_visible)
    else:
        mean_visible = torch.ones_like(start_v, device=device, dtype=dtype) / 2
        visible = start_v.to(device=device, dtype=dtype)

    return visible, mean_visible


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


def _sample_state_dmala(
    energy: torch.nn.Module,
    chains: dict[str, Tensor],
    n_steps: int,
    beta: float = 1.0,
    alpha: float = 0.25,
) -> dict[str, Tensor]:
    
    visible = chains["visible"].clone()
    weights = chains["weights"].clone()

    for _ in range(n_steps):
        visible.requires_grad_(True)

        current_energy = energy(visible).view(-1)
        grad = torch.autograd.grad(
            current_energy.sum(),
            visible,
        )[0]

        forward_logits = (
            -0.5 * beta * grad
            + (2.0 * visible - 1.0) / (2.0 * alpha)
        )

        with torch.no_grad():
            forward_prob = torch.sigmoid(forward_logits)
            proposal = torch.bernoulli(forward_prob)

        proposal_grad_input = proposal.detach().requires_grad_(True)
        proposal_energy = energy(proposal_grad_input).view(-1)
        proposal_grad = torch.autograd.grad(
            proposal_energy.sum(),
            proposal_grad_input,
        )[0]

        reverse_logits = (
            -0.5 * beta * proposal_grad
            + (2.0 * proposal_grad_input - 1.0) / (2.0 * alpha)
        )

        with torch.no_grad():
            log_q_forward = -torch.nn.functional.binary_cross_entropy_with_logits(
                forward_logits.detach(),
                proposal,
                reduction="none",
            ).sum(1)

            log_q_reverse = -torch.nn.functional.binary_cross_entropy_with_logits(
                reverse_logits.detach(),
                visible,
                reduction="none",
            ).sum(1)

            log_acceptance = (
                -beta * proposal_energy.detach()
                + beta * current_energy.detach()
                + log_q_reverse
                - log_q_forward
            )

            accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance
            visible = torch.where(accept[:, None], proposal, visible)

    return {
        "visible": visible,
        "visible_mag": visible,
        "weights": weights,
    }
