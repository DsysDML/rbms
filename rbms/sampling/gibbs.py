from torch import Tensor

from rbms.classes import RBM


def sample_state(
    gibbs_steps: int, chains: dict[str, Tensor], params: RBM, beta: float = 1.0
) -> dict[str, Tensor]:
    """Update the state of the Markov chain according to the parameters of the RBM.

    Args:
        gibbs_steps (int): Number of Gibbs steps to perform.
        chains (dict[str, Tensor]): The parallel chains used for sampling.
        params (RBM): The parameters of the RBM.
        beta (float, optional): The inverse temperature. Defaults to 1.0.

    Returns:
        dict[str, Tensor]: The updated chains after performing the Gibbs steps.
    """
    new_chains = {
        "visible": chains["visible"].clone(),
        "weights": chains["weights"].clone(),
    }
    for _ in range(gibbs_steps):
        new_chains = params.sample_hiddens(chains=new_chains, beta=beta)
        new_chains = params.sample_visibles(chains=new_chains, beta=beta)
    return new_chains
