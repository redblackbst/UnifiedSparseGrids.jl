# UnifiedSparseGrids.jl

UnifiedSparseGrids.jl is a Julia package for **regular sparse-grid data structures, transforms, and matrix-free tensor operators** built around

- nested 1D **axis families**,
- downward-closed **refinement-index sets** such as Smolyak, weighted Smolyak, and full tensor, and
- explicit coefficient layouts with deterministic sparse-grid traversal.

The package is designed for interpolation / approximation workflows and sparse-grid Galerkin experiments where you want direct control over layouts, transforms, basis changes, and matrix-free operator application.

## Installation

Until the package is registered, install it directly from GitHub:

```julia
import Pkg
Pkg.add(url="https://github.com/redblackbst/UnifiedSparseGrids.jl")
```

Once registered, `Pkg.add("UnifiedSparseGrids")` will work as usual.

## Quick start

A sparse grid is described by

- a 1D **axis family** per dimension (for example Chebyshev--Gauss--Lobatto axes), and
- a downward-closed **refinement-index set**.

These are stored in a `SparseGridSpec`, and wrapped by `SparseGrid`.

```julia
using UnifiedSparseGrids

D = 2
L = 4
axes = ntuple(_ -> ChebyshevGaussLobattoNodes(), Val(D))
I = SmolyakIndexSet(D, L)

grid = SparseGrid(SparseGridSpec(axes, I))
@show length(grid)
```

Sample a function on sparse-grid points (ordered consistently with `traverse(grid)`):

```julia
vals = evaluate(grid) do x
    exp(-sum(abs2, x))
end
```

## Available today

- Smolyak, weighted Smolyak, and full tensor refinement-index sets
- nested axis families such as Chebyshev--Gauss--Lobatto, dyadic, and Fourier equispaced rules
- recursive and subspace (block) coefficient layouts, with explicit conversions
- the unidirectional engine for hierarchization, nodal--modal transforms, basis changes, and matrix-free tensor operators
- evaluation planning on tensor grids and scattered point sets
- cross-grid transfers (`restrict!` / `embed!`) and Galerkin utilities
- worked tutorials covering sparse FFT splitting, sparse spectral Galerkin elliptic solves, and hierarchical-hat Helmholtz operators

## Planned / in progress

- more axis families and transforms:
  - Gauss / Gauss--Lobatto rules with fast tensorized transforms,
  - nested sequences such as Patterson / Leja / pseudo-Gauss,
  - dyadic wavelet / Walsh--Hadamard transforms,
  - Chebyshev--Gauss (roots) with DCT-II / DCT-III,
- combination-technique workflows,
- adaptivity (dimensional and spatial),
- sparse-grid quadrature,
- tighter PDE tooling integration, for example via [GalerkinToolkit.jl](https://github.com/GalerkinToolkit/GalerkinToolkit.jl).

## Related projects

- [SG++](https://github.com/SGpp/SGpp): sparse grids toolbox
- [Tasmanian](https://github.com/ORNL/TASMANIAN): sparse grids for UQ / interpolation
- [ASGarD](https://github.com/project-asgard/asgard): adaptive sparse grid DG for high-dimensional PDEs
- [SparseGridQuadrature.jl](https://github.com/eschnett/SparseGridQuadrature.jl): Julia sparse-grid quadrature
- [smolyax](https://github.com/JoWestermann/smolyax): Smolyak interpolation operator in JAX

## Where to start

- Coefficient ordering and iteration: [Layouts and iteration](@ref)
- Fast tensor-product sweeps (including nodal--modal transforms): [Unidirectional principle](@ref)
- Evaluating sparse-grid coefficient vectors at point sets: [Evaluation](@ref)
- Matrix-free Galerkin building blocks: [Galerkin tools](@ref)
- End-to-end examples: see the *Tutorials* section in the navigation sidebar

```@contents
Pages = [
    "manual/layouts.md",
    "manual/unidirectional_principle.md",
    "manual/evaluation.md",
    "manual/galerkin.md",
    "examples/gradinaru_2007_tdse.md",
    "examples/shen_yu_2010_sec4.md",
    "examples/balder_zenger_1996_helmholtz.md",
    "advance/conventions.md",
    "advance/development.md",
]
Depth = 2
```
