# Datasets

## Create a compatible dataset

In order to use this package, you need to provide the datasets. Only two formats are accepted so far:

- HDF5 file (`.h5`). These files are used for binary datasets. It should have a `"samples"` key inside referring to a `num_samples x num_dimension` float matrix. Optionally, a `"labels"` key can be set with a vector of numerical labels associated to the samples.
- FASTA file (`.fasta`). These files are used for proteins datasets.

An example of the code used to create a valid HDF5 dataset in python:

```python
import h5py
import numpy as np

num_samples = 100
num_dimensions = 27

data = np.random.randn(num_samples, num_dimension)

# begin optional
num_classes = 3
labels = np.random.randint(0, num_classes, size=(num_samples,))
# end optional

with h5py.File("my_dataset.h5", "w") as f:
   f["samples"] = data

   # begin optional
   f["labels"] = labels
   # end optional
```

## Command line arguments

When running a script requiring a dataset, you have several flags:

- `-d` or `--data` The path to the dataset (e.g.`-d ./data/dataset.h5`).
- `--subset_labels` For datasets with labels specified, it allows to select only a subset of the dataset matching the specified labels. For example setting `-d MNIST --subset_labels 0 1` will load the $0$ and $1$ digits of the MNIST dataset (what we refer to as `MNIST-01` dataset). If specified in a dataset without labels, the full dataset is selected.
- `--train_size` The proportion of the dataset to use as training dataset. It can go from $0$ to $1$ and the default is $0.6$.
- `--test_size` Same as above, but for the test dataset. Defaults to $1-$ train size
- `--use_weights` Compute the weights for protein sequences.
- `--alphabet` One of `{protein,rna,dna}`. Depends on the type of fasta file you are using.
