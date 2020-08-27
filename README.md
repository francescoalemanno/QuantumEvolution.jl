# QuantumEvolution.jl
Implentation of `Quantum PSO` with some modifications

[![Build Status](https://github.com/francescoalemanno/QuantumEvolution.jl/workflows/CI/badge.svg)](https://github.com/francescoalemanno/QuantumEvolution.jl/actions)

## API

```julia
function optimize(
    f,
    s,
    N;
    K = 4,
    beta = 0.95,
    max_iters = 100,
    min_cost = -Inf,
    verbose = false,
    rng = Random.GLOBAL_RNG,
    callback = state -> nothing,
)
```

- `f` : cost function to minimize, whose argument is either a scalar or a vector, must returns a scalar value.
- `s` : function whose input is the particle number and output is a random initial point to be ranked by `f`.
- `N` : number of particles to use, choose a number greater than `d+4` where `d` is the number of dimensions.
- `K` : number of particles used as best individuals to propose new particles.
- `beta` : dilation/contraction factor for proposal distribution.
- `max_iters` : maximum number of iterations.

## Usage example:

```julia
using QuantumEvolution
rosenbrock2d(x) = (x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2
initpoint(i) = randn(2)
optimize(rosenbrock2d, initpoint, 30)
```

```
(x = [1.0000000000001437, 1.0000000000002875], fx = 2.064394754776154e-26, f_nevals = 1996)
```
as expected the global optimum has been found.