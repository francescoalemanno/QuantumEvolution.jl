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
    fnevals::Int
    improv::Int
end

evals(state::QEState) = state.fnevals

struct QEProblem{tF,tS,tP,T}
    f::tF
    s::tS
    projector::tP
    beta::T
    max_iters::Int
end

function opt_state(problem::QEProblem, N, K; kw...)
    P = [problem.projector(problem.s(i)) for i = 1:N]
    C = [problem.f(P[i]) for i = 1:N]
    for i in eachindex(P)
        isfinite(C[i]) && continue
        error("The cost function has produced non-finite values for the initial point: $(P[i])")
    end
    order = sortperm(C)
    eqP = [P[i] * 1 for i in order[1:K]]
    eqC = [C[i] * 1 for i in order[1:K]]
    (eqP[1], eqC[1], QEState(P[order], C[order], eqP, eqC, N, K, 1, N, 1))
end

function opt_state(problem::QEProblem, S::QEState; rng, verbose, kw...)
    P, C, eqP, eqC, N, K, iters, fnevals, improv =
        S.P, S.C, S.eqP, S.eqC, S.N, S.K, S.iter, S.fnevals, S.improv
    cost, beta, max_iters, projector =
        problem.f, problem.beta, problem.max_iters, problem.projector
    eqPav = sum(x -> x / K, eqP)
    dsol = size(eqPav)
    tr = iters / max_iters
    t = (1 - tr)^(2tr)
    worst = C[end]
    best = C[1]

    for i = ceil(Int, (N - K + 1) * (1 - t) + t):N
        @label regen
        eqPr = (K + 1) * rand(rng) < 1 ? eqPav : rand(rng, eqP)
        log_u = log(rand(rng)) / log(2)
        v = sign.(rand(rng, dsol...) .- 0.5)
        nP = projector(@.(eqPr + beta * abs(P[i] - eqPr) * log_u * v))
        nC = cost(nP)
        fnevals += 1
        isfinite(nC) || begin
            verbose && println("Failed function evaluation at $nP")
            @goto regen
        end
        if nC <= worst
            push!(P, nP)
            push!(C, nC)
            nC < best ? improv += 1 : nothing
        end
    end
    iters += 1
    #efficiency = (improv/iters)/(fnevals/(N*0.497595+K*(1-0.497595)))
    #@show efficiency
    order = sortperm(C)[1:N]
    eqP = [P[i] * 1 for i in order[1:K]]
    eqC = [C[i] * 1 for i in order[1:K]]
    (eqP[1], eqC[1], QEState(P[order], C[order], eqP, eqC, N, K, iters, fnevals, improv))
end


"""
Implementation of `Quantum PSO` with some modifications

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
    projector = identity,
)
    N < 2K && error("N must be at least $(2K)")
    problem = QEProblem(f, s, projector, beta, max_iters)
    best, cost, state = opt_state(problem, N, K)
    callback(state) && @goto ret
    for i = 1:(max_iters)
        best, cost, state = opt_state(problem, state; rng, verbose)
        verbose &&
            println(i, " ", cost, " ", state.improv, " ", state.improv / evals(state))
        callback(state) && break
        cost < min_cost && break
    end
    @label ret
    (x = best, fx = cost, f_nevals = evals(state))
end

export optimize
end
