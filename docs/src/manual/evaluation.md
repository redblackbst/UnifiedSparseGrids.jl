# Evaluation

UnifiedSparseGrids separates

1. choosing **evaluation points** (tensor grids, scattered points), and
2. building an **evaluation plan** (reusable) that maps sparse grid coefficients to values.

## Point sets

Evaluation points are represented by `AbstractPointSet`:

- `TensorProductPoints(xs)`
- `ScatteredPoints(X)`

where `xs` is a tuple of 1D point vectors and `X` is a `d×N` matrix.

For example, a tensor grid of Chebyshev–Gauss–Lobatto points with endpoints removed
(a common choice for Dirichlet problems) in $D$ dimensions:

```julia
using UnifiedSparseGrids

D = 3
r = 6
x = points(ChebyshevGaussLobattoNodes(; endpoints=false), r)
P = TensorProductPoints(ntuple(_ -> x, Val(D)))
```

Or a scattered point cloud:

```julia
using UnifiedSparseGrids

D = 2
X = randn(D, 10_000)
P = ScatteredPoints(X)
```

## Planning and evaluating

To evaluate a sparse grid coefficient vector `u` on a point set `P`, build an evaluation plan and apply it:

```julia
plan = plan_evaluate(grid, DirichletLegendreBasis(), P, Float64)
values = evaluate(plan, u)
```

The plan can be reused to evaluate many coefficient vectors on the same point set.

Here `basis` declares how to interpret the coefficient vector `u` (e.g. Legendre modal coefficients with Dirichlet boundary conditions).

## Sampling a function on a sparse grid

For convenience, `evaluate(grid, f)` samples a function `f` on the sparse grid points
(ordered consistently with `traverse(grid)`):

```julia
vals = evaluate(grid) do x
    exp(-sum(abs2, x))
end
```

This is a common starting point for experiments and for assembling right-hand side vectors.

See the tutorials for end-to-end usage, including sparse Galerkin solves and time-dependent splitting methods.
