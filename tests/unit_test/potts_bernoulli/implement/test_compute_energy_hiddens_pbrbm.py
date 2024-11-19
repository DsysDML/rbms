import itertools

import numpy as np
import pytest
import torch

from rbms.potts_bernoulli.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _init_parameters,
)


def test_compute_energy_hiddens(sample_binary_h_samples, sample_params_pbrbm):
    # Arrange
    h, _ = sample_binary_h_samples
    vbias, hbias, weight_matrix = sample_params_pbrbm

    # Act
    energy = _compute_energy_hiddens(h, vbias, hbias, weight_matrix)
    # Assert
    assert len(energy.shape) == 1
    assert energy.shape[0] == pytest.NUM_SAMPLES


def test_exact_energy_hidden():
    NUM_STATES = 3
    NUM_VISIBLES = 2
    NUM_HIDDENS = 5
    NUM_SAMPLES = 7

    data = torch.randint(0, NUM_STATES, (NUM_SAMPLES, NUM_VISIBLES))
    vbias, hbias, weight_matrix = _init_parameters(
        num_hiddens=NUM_HIDDENS,
        data=data,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    hidden_confs = torch.bernoulli(torch.ones(NUM_SAMPLES, NUM_HIDDENS) / 2).float()

    all_visibles = torch.tensor(
        list(itertools.product(range(NUM_STATES), repeat=NUM_VISIBLES)),
        dtype=torch.float32,
    )
    tmp = torch.zeros(NUM_SAMPLES)
    all_energy = []
    for idx_vis in range(all_visibles.shape[0]):
        energy = _compute_energy(
            all_visibles[idx_vis : idx_vis + 1],
            hidden_confs,
            vbias,
            hbias,
            weight_matrix,
        )
        all_energy.append(-energy)
    tmp = torch.logsumexp(torch.vstack(all_energy), dim=0)

    energy_bis = _compute_energy_hiddens(hidden_confs, vbias, hbias, weight_matrix)
    tmp_bis = -energy_bis

    assert np.allclose(tmp, tmp_bis)
