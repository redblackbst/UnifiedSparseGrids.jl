# Conventions

This page collects the main conventions used throughout **UnifiedSparseGrids.jl**.

## Refinement indices in 1D axis families

A 1D axis family is indexed by a **refinement index** $r = 0, 1, 2, \ldots$.
For nodal axis families this defines a sequence of point sets

```math
X_0 \subset X_1 \subset X_2 \subset \cdots,
```

with incremental blocks

```math
\Delta X_r = X_r \setminus X_{r-1},\qquad r \ge 0,
```

and API accessors:

- `points(axis, r)` returns $X_r$,
- `newpoints(axis, r)` returns $\Delta X_r$,
- `totalsize(axis, r)` returns $|X_r|$,
- `blocksize(axis, r)` returns $|\Delta X_r|$.

The old names `npoints` / `delta_count` still exist for nodal axes, but
`totalsize` / `blocksize` are the preferred concepts.

### Endpoint handling at $r = 0$

Many nodal axis families expose an `endpoints` setting. In this package it is
interpreted as a **2-bit mask** describing which endpoints are kept at
$r = 0$:

- `endpoints=:none` keeps no endpoints,
- `endpoints=:left` keeps only the left endpoint,
- `endpoints=:right` keeps only the right endpoint,
- `endpoints=:both` keeps both endpoints.

Equivalently, you may pass `endpoints=false/true`, or `Val(mask)` with
$mask \in 0:3$.

With this convention, refinement index $r = 0$ is exactly the set of kept
endpoints (possibly empty), and interior refinement starts at $r = 1$.

### Local-support bases and boundary points

For local-support dyadic bases such as `HatBasis()` and
`PiecewisePolynomialBasis()`, the package follows the rule:

> Boundary degrees of freedom exist if and only if boundary points exist.

So Dirichlet-style discretizations are typically expressed by choosing an axis
family with `endpoints=:none`, not by switching to a separate “interior-only”
basis type.

## Axis families vs. index sets

A sparse grid is determined by two independent ingredients:

1. a tuple of 1D **axis families**, which define what one refinement index $r$
   means in each dimension, and
2. a downward-closed **multi-index set** $I \subset \mathbb{N}_0^D$, which decides
   which refinement vectors are active.

Therefore the approximation meaning of an index set depends on the chosen axis
families. For example, the same `FullTensorIndexSet` can represent a box in
nested nodal refinements, one-point-at-a-time Leja growth, or something else in
a future modal backend.

For iteration and planning, use `refinement_caps(grid)` (or
`refinement_caps(indexset)`) to obtain the per-dimension bounds used by the
recursive engine.

## Smolyak index sets

`SmolyakIndexSet(D, L)` is the classical isotropic Smolyak set in
refinement-index coordinates:

```math
r \in \mathbb{N}_0^D,\qquad \sum_{j=1}^D r_j \le L,\qquad r \le \mathrm{cap}.
```

Mapping to the classical 1-based notation,
$r_{\mathrm{old}} = r + 1$ and $q_{\mathrm{old}} = L + 1$ give

```math
\|r_{\mathrm{old}}\|_1 \le q_{\mathrm{old}} + D - 1.
```

A weighted variant is provided by `WeightedSmolyakIndexSet(D, L, weights)`:

```math
r \in \mathbb{N}_0^D,\qquad \sum_{j=1}^D \theta_j r_j \le L,\qquad r \le \mathrm{cap},
```

with positive integer weights $\theta_j = \mathrm{weights}[j]$.
This is the natural place to encode anisotropy such as parabolic space-time
scaling.

## Full tensor index sets

`FullTensorIndexSet(D, R)` is a refinement-index box:

```math
r \in \mathbb{N}_0^D,\qquad 0 \le r_j \le \mathrm{cap}_j.
```

If `cap` is omitted, all dimensions use the isotropic bound $R$.

## Matrix-family convention for line operators

For `LineDiagonalOp(f)` and `LineBandedOp(f)`, the user-supplied family function
must have the signature

```julia
f(n, T)
```

where:

- `n` is the 1D size of the active fiber (for example a modal count or matrix
  dimension), and
- `T` is the desired scalar element type.

The function should be a top-level function, not a closure.

## Thread-safety contract for line plans

`lineplan(op, axis, rmax, T)` returns a plan vector `planvec`, and the unidirectional engine
passes `planvec[r+1]` into `apply_line!`. Planned families are typically constructed via
`make_plan_shared(op, axis, rmax, T)` and `make_plan_entry(op, axis, n, r, T, shared)` with
`n = totalsize(axis, r)`.

The project assumes the following contract:

- Plan entries must be **read-only during application**.
- Plans must not own mutable scratch buffers (no internal `work`, `tmp`, etc.).
- All scratch memory is supplied by the caller via the `work` argument of `apply_line!`.

This enables safe multithreading: plans can be shared across threads, and only per-thread
scratch buffers are needed.
