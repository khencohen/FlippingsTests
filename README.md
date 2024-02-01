# Gaussian Flippings Tests

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Understanding the structure of datasets used to train neural networks is of great theoretical and practical importance. 
In particular, the properties of the sample covariance matrix are known in certain cases to dictate the training and generalization performance. 
In this work, we focus on binary classification of high dimensional overlapping Gaussian mixture model (GMM) data, as a tractable yet informative proxy for modeling real-world image datasets.
We derive closed-form expressions for the Bayes optimal decision boundaries under certain assumptions, showing that the boundaries are determined by the eigenstructure of the class covariances. Through experiments on synthetic GMMs, we demonstrate that deep neural networks trained for classification, learn predictors that approximate the derived optimal classifiers. 
We further extend our study to networks trained on authentic data, observing that decision thresholds correlate with the covariance eigenvectors rather than the eigenvalues, mirroring our GMM analysis. This provides theoretical insights regarding neural networks' ability to perform probabilistic inference and distill statistical patterns from intricate distributions.

In this repository we provide code for the flipping test as was implemented in our paper:

![Screenshot 2024-01-30 at 11 30 06](https://github.com/khencohen/GaussianFlippings/assets/52878011/7f59b9df-feb5-4983-9e28-cabda41e36aa)


Note: We are now working on uploading the code, and it will be ready in the near future.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)


## Installation
git clone https://github.com/khencohen/GaussianFlippings.git
cd GaussianFlippings
pip install -r requirements.txt



## Usage
python run_flipping_experiment.py



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
