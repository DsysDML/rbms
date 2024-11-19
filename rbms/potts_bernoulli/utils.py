from rbms.potts_bernoulli.classes import PBRBM


def ensure_zero_sum_gauge(params: PBRBM) -> None:
    """Ensure the weight matrix has a zero-sum gauge.

    Args:
        params (PBRBM): The parameters of the RBM.
    """
    params.weight_matrix -= params.weight_matrix.mean(1, keepdim=True)
