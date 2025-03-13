[![codecov](https://codecov.io/gh/DsysDML/rbms/graph/badge.svg?token=HRWJLZRBDD)](https://codecov.io/gh/DsysDML/rbms)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Static Badge](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-green)

# RBMs

## Overview

`rbms` is a GPU-accelerated package designed to train and analyze Restricted Boltzmann Machines (RBMs). It is intended for students and researchers who need an efficient tool for working with RBMs.
Features:

- GPU Acceleration through PyTorch.
- Multiple RBM Types: Supports Bernoulli-Bernoulli RBM and Potts-Bernoulli RBM.
- Extensible Design: Provides an abstract class RBM with methods that can be implemented for new types of RBMs, minimizing the need to reimplement training algorithms, analysis methods, and sampling methods.

## Installation

### PyPI

`rbms` can be installed from PyPI:

```bash
pip install rbms
```

The last version on PyPI corresponds to the main branch of this repo. To update the installed package run

```bash
pip install rbms --upgrade
```

### Github

If you want a more up-to-date version or wish to contribute, you can install the package directly from this repository:

```bash
git clone git@github.com:DsysDML/rbms.git
cd rbms && pip install -e .
```

If you want to update the package run

```bash
git pull
```

from inside the cloned repo.

## Dependencies

The dependencies are included in the pyproject.toml file and should be compiled in the wheel distribution. If you have a GPU, make sure to install PyTorch with GPU support.

## Usage

Main Classes and Functions:

- `EBM`: An abstract class that provides the basic structure and methods for Energy-Based Models (EBMs).
- `RBM`: An abstract class that provides the basic structure and methods for RBMs.
- `BBRBM`: A concrete implementation of the Bernoulli-Bernoulli RBM.
- `PBRBM`: A concrete implementation of the Potts-Bernoulli RBM.
- `rbms.sampling`: Submodule for sampling methods.
- `rbms.training`: Submodule for training algorithms.

## Basic Example

Train a RBM on MNIST-01 with PCD-100 for 10 000 steps, using 200 hidden nodes and 2000 permanent chains:

```bash
rbms train  -d ./data/MNIST.h5 --subset_labels 0 1 \
--num_hiddens 200 --gibbs_steps 100 --num_chains 2000 \
--num_updates 10000 --filename ./RBM_MNIST01.h5
```

## Documentation

The documentation is available at [https://dsysdml.github.io/rbms/](https://dsysdml.github.io/rbms/)

## Contributing

We welcome contributions to the development of TorchRBM. Here's how you can contribute:

- **Fork the Repository**: Fork the main repository and make your changes.
- **Propose a Merge Request**: Once your changes are complete, propose a merge request.
- **Testing**: Ensure your code is tested using the pytest framework. Include your tests in the tests folder.
- **Code Style**: Follow the pre-commit configuration for code style.
- **Documentation**: Document your code using the Google docstring style and provide type hints .
- **Dependencies**: Avoid adding extra dependencies unless absolutely necessary.

## Support

If you encounter any issues, please open an issue on the main repository.

## License

`rbms` is released under the MIT License.
