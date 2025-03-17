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

## [What's New](whats_new.rst)

### Version 0.2

- EBM abstract class for more generic EBMs.
-

See [Releases](https://github.com/DsysDML/rbms/releases)

## [Restricted Boltzmann Machines](rbm.md)

## [Example gallery](auto_examples/index.rst)

## [Tutorials](tutorials.md)

## [API](api.md)

```{toctree}
:caption: 'Contents:'
:maxdepth: 4
```
