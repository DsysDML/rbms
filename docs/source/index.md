 # Welcome to Restricted Boltzmann Machines (RBM) in PyTorch 

`rbms` is a GPU-accelerated package designed to train and analyze Restricted Boltzmann Machines (RBMs). It is intended for students and researchers who need an efficient tool for working with RBMs.

## Features

- **GPU Acceleration**.
- **Multiple RBM Types**: Supports Bernoulli-Bernoulli RBM and Potts-Bernoulli RBM.
- **Extensible Design**: Provides an abstract class `RBM` with methods that can be implemented for new types of RBMs, minimizing the need to reimplement training algorithms, analysis methods, and sampling methods.


## Installation

To install `rbms`, you can use pip:

```bash
pip install rbms
```

## What's New

### Version 0.1.1
 - Fix an error biasing the estimation of the LL using Annealed Importance Sampling

### Version 0.1

- Initial release of TorchRBM.
- Support for Bernoulli-Bernoulli RBM and Potts-Bernoulli RBM.
- Abstract class RBM for easy extension to new RBM types.
- GPU acceleration for training and analysis.
- Comprehensive sampling and training submodules.

## [Restricted Boltzmann Machines](rbm.md)

## [Example gallery](auto_examples/index.rst)

## [Tutorials](tutorials.md)

## [API](api.md)



```{toctree}
:caption: 'Contents:'
:maxdepth: 4
```

