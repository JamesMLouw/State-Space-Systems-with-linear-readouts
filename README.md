# Learning the climate of dynamical systems with state-space systems
"Learning the climate of dynamical systems with state-space systems"
J. Louw, and J.P. Ortega
*Arxiv* arXiv:2512.15530
(https://arxiv.org/abs/2512.15530)

SOURCE
------
Code base comes from https://github.com/Learning-of-Dynamic-Processes/coldstart.git
"Data-driven cold starting of good reservoirs"
L. Grigoryeva et al., (2023).
*ArXiv* arXiv:2403.10325 
(https://arxiv.org/abs/2403.10325)

INSTALLATION
---------


Clone repository and install dependencies

    git clone https://github.com/Learning-of-Dynamic-Processes/Learning-the-climate-with-state-space-systems.git

USAGE
---------

- Run `lorenz_optuna.py` to find a good set of reservoir hyperparamerters for the Lorenz system. Paste the best set of parameters into `lorenz/config.py`.
- Run `train_model.py` to generate training and testing data and to train the ESN on the training data.
- Run `predict_model.py` to use the ESN to predict from the testing data. Includes code to generate the figures used in the paper.
