from rbms.potts_bernoulli.classes import PBRBM


def ensure_zero_sum_gauge(params: PBRBM) -> None:
    """Ensure the weight matrix has a zero-sum gauge.

    Args:
        params (PBRBM): The parameters of the RBM.
    """
    mean_W = params.weight_matrix.mean(1, keepdim=True)
    params.weight_matrix -= mean_W
    params.hbias += mean_W.squeeze().sum(0)
    params.vbias -= params.vbias.mean(1, keepdim=True)
