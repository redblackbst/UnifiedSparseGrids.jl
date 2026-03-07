# Galerkin tools

UnifiedSparseGrids provides a small set of building blocks for **matrix-free** sparse grid Galerkin workflows.
They are designed to work with coefficients stored in the package's recursive layout and to compose efficiently
with the [Unidirectional principle](@ref).

## Matrix-free tensor-sum operators

Many PDE operators can be written as a *sum of separable tensor terms*, for example

```math
(-\Delta + \alpha) = \sum_{j=1}^D I \otimes \cdots \otimes K_j \otimes \cdots \otimes I
\; + \; \alpha\,(I \otimes \cdots \otimes I),
```

where each `K_j` is a 1D stiffness-like operator in dimension `j`.

The generic helper types are:

- `WeightedTensorTerm(w, op)` for a weighted `AbstractTensorOp` term,
- `TensorSumMatVec(grid, terms, T)` for a `mul!`-compatible matrix-free matvec applying a sum of terms,
- `as_linear_operator(A)` to wrap a `TensorSumMatVec` as a `LinearOperators.LinearOperator` for Krylov solvers.

These are used throughout the examples to define operators without forming large dense matrices.

## Hat-basis line operators (Balder–Zenger)

For the hierarchical hat basis used in the Balder–Zenger Helmholtz example, the package provides
specialized 1D building blocks:

- `HatMassLineOp` (with `updown(op)` support for triangular splitting),
- `HatStiffnessDiagLineOp`.

The key design point is that these line operators are **matrix-free and in-place**:
they implement `apply_line!` for a single fiber and are composed into a tensor operator via `TensorOp`/
`UpDownTensorOp`.

## Discrete error utilities

Examples often validate solutions by comparing discrete samples against a reference function `uex`.
This is supported by `discrete_l2_error`, which is available at three convenience levels:

- **High-level**: `discrete_l2_error(u, grid, basis, uex, ref_level; eval_nodes=...)`.
  Builds a tensor-product reference point set and evaluates the sparse grid expansion.
- **Medium-level**: `discrete_l2_error(u, grid, basis, uex, eval_points)`.
  Evaluates via the evaluation interface (`plan_evaluate` + `evaluate`).
- **Low-level**: `discrete_l2_error(uvals, uex, eval_points)`.
  Here `uvals` is interpreted as already aligned with `eval_points`.

All variants return `(L2, relL2)` where `L2` is the root-mean-square error and `relL2` is the relative RMS error.
