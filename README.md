[![codecov](https://codecov.io/gh/DsysDML/test_rbm_workflows/graph/badge.svg?token=QS0Q8BJO09)](https://codecov.io/gh/DsysDML/test_rbm_workflows) 
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Static Badge](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-green)


# TorchRBM

## Overview

TorchRBM is a GPU-accelerated package designed to train and analyze Restricted Boltzmann Machines (RBMs). It is intended for students and researchers who need an efficient tool for working with RBMs.
Features:

 - GPU Acceleration through PyTorch.
 - Multiple RBM Types: Supports Bernoulli-Bernoulli RBM and Potts-Bernoulli RBM.
 - Extensible Design: Provides an abstract class RBM with methods that can be implemented for new types of RBMs, minimizing the need to reimplement training algorithms, analysis methods, and sampling methods.

## Installation

To install TorchRBM, you can use pip:

```bash
pip install torchrbm
```

## Dependencies

The dependencies are included in the pyproject.toml file and should be compiled in the wheel distribution. If you have a GPU, make sure to install PyTorch with GPU support.

## Usage

Main Classes and Functions:
 - RBM: An abstract class that provides the basic structure and methods for RBMs.
 - BBRBM: A concrete implementation of the Bernoulli-Bernoulli RBM.
 - PBRBM: A concrete implementation of the Potts-Bernoulli RBM.
 - torchrbm.sampling: Submodule for sampling methods.
 - torchrbm.training: Submodule for training algorithms.

## Basic Example

(We will add a basic example here later.)
Examples and Tutorials

(We will add specific examples and tutorials here later.)
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
TorchRBM is released under the MIT License.