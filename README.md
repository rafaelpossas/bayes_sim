# Bayes Sim - Adaptive domain randomization via probabilistic inference for robotics simulators

Implementation of Bayes Sim using Mixture of Density Network and Mixture of Density Random Fourier Features

### Introduction

We introduce BayesSim, a framework for robotics
simulations allowing a full Bayesian treatment for the parameters
of the simulator. As simulators become more sophisticated and
able to represent the dynamics more accurately, fundamental
problems in robotics such as motion planning and perception can
be solved in simulation and solutions transferred to the physical
robot. However, even the most complex simulator might still not
be able to represent reality in all its details either due to inac-
curate parametrization or simplistic assumptions in the dynamic
models. BayesSim provides a principled framework to reason
about the uncertainty of simulation parameters. Given a black
box simulator (or generative model) that outputs trajectories of
state and action pairs from unknown simulation parameters, fol-
lowed by trajectories obtained with a physical robot, we develop
a likelihood-free inference method that computes the posterior
distribution of simulation parameters. This posterior can then be
used in problems where Sim2Real is critical, for example in policy
search. We compare the performance of BayesSim in obtaining
accurate posteriors in a number of classical control and robotics
problems, and show that the posterior computed from BayesSim
can be used for domain radomization outperforming alternative
methods that randomize based on uniform priors.

### License

The content of this repository is licensed under the Creative Commons Attribution Share Alike 4.0 License.

### Installation

1. Install [Tensorflow](https://www.tensorflow.org/).

2. Install [DELFI](https://github.com/mackelab/delfi) 0.5.1

3. Install [OpenAI-Gym](https://github.com/openai/gym)

4. Install [Stable-Baselines](https://stable-baselines.readthedocs.io/en/master/guide/install.html)

4. Install Numpy and SKLearn


### Required environment
- Ubuntu 16.04
- Python 3.6
- CUDA 9.1

### Paper:
```
@inproceedings{ramos2019rss:bayessim,
 author = {Fabio Ramos and Rafael Possas and Dieter Fox},
 title = {BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators},
 booktitle = {Robotics: Science and Systems (RSS)},
 url = "https://arxiv.org/abs/1906.01728",
 year = 2019
}
```

### Running the demo
1. Run the pendulum_param_inference.py in the root directory for a simple example
