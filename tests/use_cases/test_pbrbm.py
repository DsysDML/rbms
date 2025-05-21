import torch

from rbms.dataset import load_dataset
from rbms.io import load_model, load_params
from rbms.partition_function.exact import compute_partition_function
from rbms.scripts.train_rbm import train_rbm
from rbms.utils import (
    compute_log_likelihood,
    get_categorical_configurations,
)

# We are testing here the training of a dummy RBM with a full batch gradient and
# a few hidden nodes in order to compute the true LL. We perform a few gradient updates
# and if the train ll doesn't go up then there is a problem and the test fails.
# It should be the whole pipeline running by just calling the uppermost function and computing the
# LL a posteriori.


def test_use_case_train_pbrbm_no_weights():
    # Generate dummy dataset
    # NUM_SAMPLES = 2270
    NUM_CHAINS = 223
    # NUM_VISIBLES = 805
    NUM_HIDDENS = 5
    LEARNING_RATE = 0.01
    GIBBS_STEPS = 101
    NUM_UPDATES = 100
    SUBSET_LABELS = [0, 1]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32

    BATCH_SIZE = 300
    filename = "RBM.h5"

    args = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_chains": NUM_CHAINS,
        "num_hiddens": NUM_HIDDENS,
        "gibbs_steps": GIBBS_STEPS,
        "log": False,
        "device": DEVICE,
        "dtype": DTYPE,
        "num_updates": NUM_UPDATES,
        "filename": filename,
        "beta": 1.0,
        "restore": False,
        "data": "dummy.fasta",
        "subset_labels": SUBSET_LABELS,
        "use_weights": False,
        "alphabet": "protein",
        "train_size": 0.6,
        "test_size": None,
        "n_save": 50,
        "spacing": "exp",
        "binarize": False,
        "overwrite": True,
        "seed": 42,
        "no_center": False,
        "L1": 0.0,
        "L2": 1.0,
    }
    train_rbm(args)

    train_dataset, test_dataset = load_dataset(
        "dummy.fasta",
        subset_labels=SUBSET_LABELS,
        use_weights=args["use_weights"],
        binarize=args["binarize"],
        device=DEVICE,
    )

    params, chains, train_time, hyperparameters = load_model(
        filename, NUM_UPDATES, DEVICE, DTYPE, restore=False
    )
    assert train_time > 0
    all_config = get_categorical_configurations(
        n_states=2, n_dim=NUM_HIDDENS, device=DEVICE, dtype=DTYPE
    )
    log_z_end = compute_partition_function(params, all_config)
    ll_train_end = compute_log_likelihood(
        train_dataset.data, train_dataset.weights, params, log_z_end
    )
    params = load_params(filename, 1, DEVICE, DTYPE)
    log_z_begin = compute_partition_function(params, all_config)

    ll_train_begin = compute_log_likelihood(
        train_dataset.data, train_dataset.weights, params, log_z_begin
    )

    # assert ll_train_end > ll_train_begin


def test_use_case_train_pbrbm_weights():
    # Generate dummy dataset
    # NUM_SAMPLES = 2270
    NUM_CHAINS = 223
    # NUM_VISIBLES = 805
    NUM_HIDDENS = 5
    LEARNING_RATE = 0.01
    GIBBS_STEPS = 101
    NUM_UPDATES = 100
    SUBSET_LABELS = [0, 1]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32

    BATCH_SIZE = 300
    filename = "RBM.h5"

    args = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_chains": NUM_CHAINS,
        "num_hiddens": NUM_HIDDENS,
        "gibbs_steps": GIBBS_STEPS,
        "log": False,
        "device": DEVICE,
        "dtype": DTYPE,
        "num_updates": NUM_UPDATES,
        "filename": filename,
        "beta": 1.0,
        "restore": False,
        "data": "dummy.fasta",
        "subset_labels": SUBSET_LABELS,
        "use_weights": True,
        "alphabet": "protein",
        "train_size": 0.6,
        "test_size": None,
        "n_save": 50,
        "spacing": "exp",
        "binarize": False,
        "overwrite": True,
        "seed": 42,
        "no_center": False,
        "L1": 1.0,
        "L2": 0.0,
    }
    train_rbm(args)

    train_dataset, test_dataset = load_dataset(
        "dummy.fasta",
        subset_labels=SUBSET_LABELS,
        use_weights=args["use_weights"],
        binarize=args["binarize"],
        device=DEVICE,
    )

    params, chains, train_time, hyperparameters = load_model(
        filename, NUM_UPDATES, DEVICE, DTYPE, restore=False
    )
    assert train_time > 0
    all_config = get_categorical_configurations(
        n_states=2, n_dim=NUM_HIDDENS, device=DEVICE, dtype=DTYPE
    )
    log_z_end = compute_partition_function(params, all_config)
    ll_train_end = compute_log_likelihood(
        train_dataset.data, train_dataset.weights, params, log_z_end
    )
    params = load_params(filename, 1, DEVICE, DTYPE)
    log_z_begin = compute_partition_function(params, all_config)

    ll_train_begin = compute_log_likelihood(
        train_dataset.data, train_dataset.weights, params, log_z_begin
    )

    assert ll_train_end > ll_train_begin
