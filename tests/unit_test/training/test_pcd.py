import numpy as np
import pytest
import torch

from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.io import load_params
from rbms.map_model import map_model
from rbms.potts_bernoulli.classes import PBRBM
from rbms.training.pcd import train


def test_train_bbrbm(sample_dataset_bbrbm, sample_args):
    dataset = sample_dataset_bbrbm
    test_dataset = sample_dataset_bbrbm
    checkpoints = np.arange(1, sample_args["num_updates"] + 1)
    sample_args["restore"] = False
    sample_args["batch_size"] = pytest.NUM_SAMPLES
    model_type = "BBRBM"
    train(
        dataset,
        test_dataset,
        model_type,
        sample_args,
        torch.float32,
        checkpoints,
        map_model=map_model,
    )

    params_begin = load_params(
        sample_args["filename"],
        index=1,
        device=sample_args["device"],
        dtype=sample_args["dtype"],
    )
    assert isinstance(params_begin, BBRBM)

    params_end = load_params(
        sample_args["filename"],
        index=sample_args["num_updates"],
        device=sample_args["device"],
        dtype=sample_args["dtype"],
    )
    assert isinstance(params_end, BBRBM)
    for k in params_begin.named_parameters().keys():
        assert not (
            torch.allclose(
                params_begin.named_parameters()[k], params_end.named_parameters()[k]
            )
        )
        assert (
            params_begin.named_parameters()[k].shape
            == params_end.named_parameters()[k].shape
        )


def test_train_pbrbm(sample_dataset_pbrbm, sample_args):
    dataset = sample_dataset_pbrbm
    test_dataset = sample_dataset_pbrbm
    checkpoints = np.arange(1, sample_args["num_updates"] + 1)
    sample_args["restore"] = False
    sample_args["batch_size"] = pytest.NUM_SAMPLES
    model_type = "PBRBM"
    train(
        dataset,
        test_dataset,
        model_type,
        sample_args,
        torch.float32,
        checkpoints,
        map_model=map_model,
    )

    params_begin = load_params(
        sample_args["filename"],
        index=1,
        device=sample_args["device"],
        dtype=sample_args["dtype"],
    )
    assert isinstance(params_begin, PBRBM)

    params_end = load_params(
        sample_args["filename"],
        index=sample_args["num_updates"],
        device=sample_args["device"],
        dtype=sample_args["dtype"],
    )
    assert isinstance(params_end, PBRBM)
    for k in params_begin.named_parameters().keys():
        assert not (
            torch.allclose(
                params_begin.named_parameters()[k], params_end.named_parameters()[k]
            )
        )
        assert (
            params_begin.named_parameters()[k].shape
            == params_end.named_parameters()[k].shape
        )
