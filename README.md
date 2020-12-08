# ProxSGD
This is the PyTorch implementation of the ProxSGD algorithm proposed in following paper:

"ProxSGD: Training Structured Neural Networks under Regularization and Constraints," **International Conference on Learning Representation (ICLR)**, Apr. 2020. [URL](https://openreview.net/forum?id=HygpthEtvr)

by *Yang Yang, Yaxiong Yuan, Avraan Chatzimichailidis, Ruud JG van Sloun, Lei Lei, and Symeon Chatzinotas.*

This repository contains the simulations in Section 4.2.

# Hyperparameter Search
This branch is created for hyperparameter search using Covariance Matrix Adaptation evolution strategy..

## Setup
Kindly follow the following steps to get experimental results.

1. Install optuna in pytorch enviornment.
2. Update config.py with required settings.
3. Run main_search_cma.py 
4. Results can be seen in directory mentioned in config.py


