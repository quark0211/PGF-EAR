# Efficiency, accuracy and robustness of probability generating function based parameter inference method for stochastic biochemical reactions

This repository contains the Julia code for the paper **"Efficiency, accuracy and robustness of probability generating function based parameter inference method for stochastic biochemical reactions"**.

## Requirements

- Julia 1.11.1

Packages:

- ApproxBayes 0.3.2
- CSV 0.10.15
- DataFrames 1.8.1
- Distributions 0.25.123
- FastGaussQuadrature 1.1.0
- HypergeometricFunctions 0.3.28
- JSON 1.4.0
- LinearAlgebra 1.11.0
- NLsolve 4.5.1
- OptimalTransport 0.3.20
- Optim 1.13.3
- OrdinaryDiffEq 6.108.0
- Plots 1.41.4
- SparseArrays 1.11.0
- Statistics 1.11.1
- StatsBase 0.34.10
- TaylorSeries 0.20.10
- 
## File description

This repository is organized into three main parts: **data generation**, **parameter inference**, and **model selection**.  
The data generation scripts are used to simulate steady-state or time-dependent stochastic biochemical reaction data.  
The inference scripts implement and compare different parameter inference methods under steady-state and transient settings, including **PGF**, **FSP**, **MOM**, and **ABC**.  
The model selection script implements a **PGF-based transient model selection method**.

### 1. Data generation

These scripts are used to generate synthetic datasets for subsequent inference and evaluation.

- `generate_steady_data.jl`  
  Generates simulated **steady-state** data for stochastic biochemical reaction systems.  
  This script is used to construct benchmark datasets under stationary conditions for testing and comparing different steady-state inference methods.

- `generate_time_data.jl`  
  Generates simulated **time-dependent (transient)** data for stochastic biochemical reaction systems.  
  This script is used to produce temporal datasets for transient parameter inference and time-dependent model selection tasks.

### 2. Parameter inference

These scripts implement parameter inference algorithms under either steady-state or transient settings.

#### 2.1 Steady-state inference

These scripts are designed for parameter inference from steady-state data and for comparing the performance of different methods under stationary conditions.

- `ABC-steady.jl`  
  Implements the **Approximate Bayesian Computation (ABC)** method for steady-state parameter inference.  
  This script can be used to estimate model parameters from steady-state data without explicitly evaluating the likelihood function.

- `FSP-steady.jl`  
  Implements the **Finite State Projection (FSP)** method for steady-state parameter inference.  
  This script solves or approximates the steady-state probability distribution and performs parameter estimation based on the FSP framework.

- `PGF&MOM-steady.jl`  
  Implements **Probability Generating Function (PGF)**-based and **Moment (MOM)**-based methods for steady-state parameter inference.  
  This script is used to estimate parameters under stationary conditions and to compare PGF and MOM approaches in terms of efficiency, accuracy, and robustness.

#### 2.2 Time-dependent inference

These scripts are designed for parameter inference from transient data and for comparing methods under time-evolving conditions.

- `PGF&FSP-time.jl`  
  Implements **PGF-based** and **FSP-based** methods for **time-dependent** parameter inference.  
  This script is used to estimate parameters from transient stochastic data and to compare PGF and FSP approaches in dynamic settings.

- `MOM-time.jl`  
  Implements the **Moment (MOM)** method for **time-dependent** parameter inference.  
  This script is used to infer model parameters from transient data based on moment dynamics.

### 3. Model selection

This part focuses on identifying the most appropriate stochastic biochemical reaction model from time-dependent data.

- `model-selection.jl`  
  Implements the **PGF-based model selection algorithm** for **time-dependent** stochastic biochemical reaction systems.  
  This script is used to distinguish among candidate models using transient data and to evaluate the effectiveness of the PGF-based selection strategy.


## Paper link

- [Efficiency, accuracy and robustness of probability generating function based parameter inference method for stochastic biochemical reactions](https://www.biorxiv.org/content/10.64898/2026.01.21.700833v1)
