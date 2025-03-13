from torch import Tensor

from rbms.classes import EBM


def sample_state(
    gibbs_steps: int, chains: dict[str, Tensor], params: EBM, beta: float = 1.0
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
    return params.sample_state(n_steps=gibbs_steps, chains=chains, beta=beta)
