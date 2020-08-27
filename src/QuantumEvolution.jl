module QuantumEvolution
using Random, StatsBase, LinearAlgebra

struct QEState{vP,vC,vO}
    P::Vector{vP}
    C::Vector{vC}
    eqP::Vector{vO}
    eqC::Vector{vC}
    N::Int
    K::Int
    iter::Int
end

struct QEProblem{tF,tS,T}
    f::tF
    s::tS
    beta::T
    max_iters::Int
end

function opt_state(problem::QEProblem, N, K; kw...)
    P = [problem.s(i) for i = 1:N]
    C = [problem.f(P[i]) for i = 1:N]
    for i in eachindex(P)
        isfinite(C[i]) && continue
        error("The cost function has produced non-finite values for the initial point: $(P[i])")
    end
    order = sortperm(C)
    eqP = [P[i] * 1 for i in order[1:K]]
    eqC = [C[i] * 1 for i in order[1:K]]
    eqP[1], eqC[1], QEState(P[order], C[order], eqP, eqC, N, K, 1), N
end

function opt_state(problem::QEProblem, S::QEState; rng, verbose, kw...)
    P, C, eqP, eqC, N, K, iters = S.P, S.C, S.eqP, S.eqC, S.N, S.K, S.iter
    cost, beta, max_iters = problem.f, problem.beta, problem.max_iters
    eqPav = sum(x -> x / K, eqP)
    dsol = size(eqPav)
    tr = iters / max_iters
    t = (1 - tr)^(tr)
    fnevals = 0
    ndims = 1.0 * length(eqPav)
    # this dampening factor F, serves to account for multiple dimensions, 
    # as ndims increses this factor keeps the probability of sampling near the optimum constant
    F = log(2) / (2 * ndims) + log(ndims) + log(1 / log(2)) # approx of -log(1-2^(-1/ndims)), error below 3%
    for i = ceil(Int, (N - K + 1) * (1 - t) + t):N
        @label regen
        eqPr = (K + 1) * rand(rng) < 1 ? eqPav : rand(rng, eqP)
        u = rand(rng)
        v = sign.(rand(rng, dsol...) .- 0.5)
        nP = @.(eqPr + (beta / F) * abs(P[i] - eqPr) * log(u) * v)
        nC = cost(nP)
        fnevals += 1
        isfinite(nC) || begin
            verbose && println("Failed function evaluation at $nP")
            @goto regen
        end
        push!(P, nP)
        push!(C, nC)
    end
    order = sortperm(C)
    eqP = [P[i] * 1 for i in order[1:K]]
    eqC = [C[i] * 1 for i in order[1:K]]
    eqP[1],
    eqC[1],
    QEState(P[order[1:N]], C[order[1:N]], eqP, eqC, N, K, iters + 1),
    fnevals
end


"""
Implentation of `Quantum PSO` with some modifications

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
"""
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
    N < 2K && error("N must be at least $(2K)")
    problem = QEProblem(f, s, beta, max_iters)
    best, cost, state, evals = opt_state(problem, N, K)
    callback(state)
    fnevals = evals
    for i = 1:(max_iters)
        best, cost, state, evals = opt_state(problem, state; rng, verbose)
        fnevals += evals
        callback(state)
        verbose && println(i, " ", cost)
        cost < min_cost && break
    end
    (x = best, fx = cost, f_nevals = fnevals)
end

export optimize
end
