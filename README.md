# Classifying Overlapping Gaussian Mixtures in High Dimensions: From Optimal Classifiers to Neural Networks

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

We derive closed-form expressions for the Bayes optimal decision boundaries in binary classification of high dimensional overlapping Gaussian mixture model (GMM) data, and show how they depend on the eigenstructure of the class covariances, for particularly interesting structured data.
We empirically demonstrate, through experiments on synthetic GMMs inspired by real-world data, that deep neural networks trained for classification, learn predictors which approximate the derived optimal classifiers. 
We further extend our study to networks trained on authentic data, observing that decision thresholds correlate with the covariance eigenvectors rather than the eigenvalues, mirroring our GMM analysis. This provides theoretical insights regarding neural networks' ability to perform probabilistic inference and distill statistical patterns from intricate distributions.

In this repository we provide the code for tests in the paper:

![Screenshot 2024-01-30 at 11 30 06](https://github.com/khencohen/GaussianFlippings/assets/52878011/7f59b9df-feb5-4983-9e28-cabda41e36aa)



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)


## Installation
git clone https://github.com/khencohen/FlippingsTests.git

cd FlippingsTests

pip install -r requirements.txt



## Usage
python main.py



## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yourPaperID,
  title={Your Paper Title},
  author={Your Name and Co-Authors},
  journal={Journal Name or Conference},
  year={Year},
  doi={YourDOI}
}
