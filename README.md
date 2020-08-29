# QuantumEvolution.jl
Implementation of Quantum PSO with some modifications, inspired by `Species-based Quantum Particle Swarm Optimization for economic load-dispatch`

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
    projector = identity
)
```

- `f` : cost function to minimize, whose argument is either a scalar or a vector, must returns a scalar value.
- `s` : function whose input is the particle number and output is a random initial point to be ranked by `f`.
- `N` : number of particles to use, choose a number greater than `d+4` where `d` is the number of dimensions.
- `K` : number of particles used as best individuals to propose new particles.
- `beta` : dilation/contraction factor for proposal distribution.
- `max_iters` : maximum number of iterations.
- `min_cost` : threshold for stopping the algorithm whenever the cost of the current best solution is below `min_cost`
- `verbose` : enables verbosity
- `rng` : random number generator
- `callback` : callback function to access the internal state of the optimizer, if it returns `true` the optimizer is stopped
- `projector` : function whose role is projecting parameters back into their constrained space

## Usage example:

```julia
using QuantumEvolution
rosenbrock2d(x) = (x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2
initpoint(i) = randn(2)
optimize(rosenbrock2d, initpoint, 30)
```

```
(x = [0.9999998740708989, 0.9999997219238367], fx = 8.459636979174772e-14, f_nevals = 1661)
```
as expected the global optimum has been found.