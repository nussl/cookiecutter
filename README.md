Introduction
====================

A boilerplate for reproducible and transparent computer audition research that leverages
[nussl](https://interactiveaudiolab.github.io/nussl/), a source separation library. This 
project provides boilerplate code for training neural network models to separate mixtures
containing multiple speakers, music, and environmental sounds. It is easy to add and
train new models or datasets (GPU sold separately). The goal of this project is to enable
further reproducibility within the source separation community.

Usage
-----
To start a new project:

`cookiecutter gh:pseeth/cookiecutter-nussl` 

Documentation
-------------

The documentation is [here](https://pseeth.github.io/cookiecutter-nussl/). It includes
guides for getting started, training models, creating datasets, and API documentation.

Features
------------

The following models can be trained:

- Mask Inference
- Deep Clustering
- Chimera
- TasNet

on the following datasets:

- MUSDB18
- MIR-1k
- Slakh
- WSJ0-mix2
- Wham!

This project utilizes building block components from `nussl` for input/output 
(reading/writing audio, STFT/iSTFT, masking, etc.), and for neural network construction
(recurrent networks, convolutional networks, etc) to train models with minimal setup.
The main source separation library, `nussl`, contains many pre-trained models trained
using this code. See the [External File Zoo (EFZ)](http://nussl.ci.northwestern.edu/)
for trained models.

This project uses
[cookiecutter](https://cookiecutter.readthedocs.io/en/latest/readme.html).
Cookiecutter is a *logical, reasonably standardized, but flexible project structure
for doing and sharing research.* This project and `nussl` are both built upon
the [PyTorch](https://pytorch.org/) machine learning framework, as such, building new
components is as simple as adding new PyTorch code, though writing python is not required.

Requirements
------------
- Install `cookiecutter` command line: `pip install cookiecutter` (generates boilerplate 
code)

License
-------
This project is licensed under the terms of the [MIT License](/LICENSE)