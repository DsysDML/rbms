import gzip
import pickle

import h5py
import numpy as np


def preprocess_MNIST(
     filename="mnist.pkl.gz",
     binary_threshold=0.3,
     out_dir="dataset",
):
     with gzip.open(filename, "rb") as f:
         training_data, validation_data, test_data = pickle.load(f,
encoding="latin1")

     names = ["MNIST_train.h5", "MNIST_val.h5", "MNIST_test.h5"]
     datasets = [training_data, validation_data, test_data]
     for dataset, name in zip(datasets, names):
         curr_data = np.array(dataset[0])
         curr_data = (curr_data > binary_threshold).astype("float")
         curr_labels = np.array(dataset[1])

         with h5py.File(name, "w") as f:
             f["samples"] = curr_data
             f["labels"] = curr_labels


if __name__ == "__main__":
     preprocess_MNIST()