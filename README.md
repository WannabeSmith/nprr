# NPRR: Nonparametric randomized response

This code implements the nonparametric randomized response (NPRR) mechanism as well as methods for computing locally differentially private confidence intervals and sequences from NPRR's output. The methods are based on the paper ["A nonparametric extension of randomized response for private confidence sets"](https://arxiv.org/abs/2202.08728) by Ian Waudby-Smith, Zhiwei Steven Wu, and Aaditya Ramdas (2023).

## Installation

** This won't work yet, we're going to put it on PyPi soon. **

To install the package, run the following in a terminal: 

```sh
pip install nprr
```

## Organization

The package is organized into two main submodules:

- `nprr.mechanisms` implements privacy mechanisms including the NPRR and Laplace mechanisms along with utilities for working with them.
- `nprr.dpcs` implements the confidence intervals and sequences from the paper as well as utilities for working with them.
  
In addition, the package contains the following submodules:

- `nprr.cgf` implements some cumulant generating functions (CGF) and CGF-like functions that are used throughout the `nprr.dpcs` module.
- `nprr.plotting` implements functions for producing the plots found in the paper.
- `nprr.types` implements some basic type aliases used throughout the package.

## Reproducing the plots from the paper

Please begin the following steps from the root of the nprr directory.

1. Create and activate a virtual environment using the method of your choice. We used venv, e.g.

```sh
python3.9 -m venv venv_dpconc

source venv_dpconc/bin/activate
```

2. Install the package and its dependencies.

```sh
python install -e ./
```

3. Enter the plots directory.

```sh
cd plots
```

4. Generate figures.

```sh
### Figure 1 ###
python hoeffding_eps.py
# -> output file: bounded_beta_1_1_hoeffding_eps.pdf

### Figure 3 ###
python hoeffding_tightness_max.py
# -> output file: bounded_beta_50_50_tightness_max.pdf

### Figure 4 ###
python confint_bounded.py
# -> output file: bounded_beta_50_50_ci.pdf

### Figure 5 ###
python confseq_bounded.py
# -> output file: bounded_beta_50_50_cs.pdf

### Figure 6 ###
python two-sided-running-mean.py
# -> output file: wavy_cs.pdf

### Figure 7 ###
python ab-test.py
# -> output file: wavy_ipw_cs.pdf
```


