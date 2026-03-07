# Helmholtz on sparse grids in a hierarchical hat basis (Balder–Zenger 1996)

This example mirrors the sparse grid Galerkin discretization and the **fast matrix–vector product**
ideas from:

- Balder and Zenger (1996), *The Solution of Multidimensional Real Helmholtz Equations on Sparse Grids*.

The corresponding implementation is in `examples/balder_zenger_1996_helmholtz.jl`.

## Model problem

Balder–Zenger consider the *real Helmholtz* problem on the unit hypercube

```math
\begin{aligned}
-\Delta u + c\,u &= f && \text{in } \Omega=(0,1)^d,\\
 u &= 0 && \text{on } \partial\Omega,
\end{aligned}
```

with a constant reaction coefficient $c$.

In the Galerkin form (hat basis functions $\{\varphi_i\}$), the linear system is

```math
(S + c\,M)\,u = b,
```

where $S$ is the stiffness matrix for $-\Delta$ and $M$ is the mass matrix.

## Discretization ingredients used in the example

### Sparse grid

We use a Smolyak-type sparse grid index set (`SmolyakIndexSet`) with dyadic nested 1D rules
(`DyadicNodes()`). The `LevelOrder()` order is important: it stores each refinement level
as an *append-only* extension of the previous one (nested-prefix property), which is what the
recursive layout and the unidirectional engine rely on.

### 1D hierarchical hat basis

On each 1D dyadic grid level $\ell$, the basis functions are the standard piecewise-linear "hat"
functions, arranged in a hierarchical (multi-level) basis.

Two 1D operators are crucial:

- A *diagonal* stiffness operator in the hierarchical hat basis.
- A structured mass operator that can be applied in *linear* complexity.

#### 1D stiffness is diagonal

Balder–Zenger show that, in the hierarchical hat basis, the stiffness matrix is diagonal
(their Eq. (9)–(10)). In our implementation this is provided by

- `HatStiffnessDiagLineOp(nodes, rmax)`.

Because it is diagonal, `updown(op)` is trivial and the operator is naturally matrix-free.

#### 1D mass has a fast (matrix-free) apply

The 1D mass matrix is not diagonal, but it has a structure that enables an $\mathcal{O}(N)$ matvec.
Balder–Zenger derive the decomposition (their Eq. (15))

```math
M = D\,(Y^{-1}-E) + (Y^{-\mathsf{T}}-E)\,D + \tfrac{2}{3}D,
```

where `D` is diagonal (level-dependent weights), `Y` represents a hierarchical cumulative-sum operator,
and `E` is a simple rank-1 correction.

In `UnifiedSparseGrids` this is encoded as a **matrix-free, in-place** line operator

- `HatMassLineOp(nodes, rmax; part=:full)`

that stores only the diagonal weights plus a scratch buffer, and evaluates the action of `M` using
hierarchization/dehierarchization-style sweeps.

A key design choice in this package is that sparse grid tensor operators are applied without ever
materializing intermediate data outside the sparse index set.

To do this safely for operators that are *not* block-diagonal on the sparse index set, we provide

- `updown(op::HatMassLineOp{:full}) -> (HatMassLineOp{:lower}, HatMassLineOp{:upper})`

so that `HatMassLineOp` can be used inside `UpDownTensorOp`.

## Multidimensional operator as a tensor sum

Let $M$ denote the 1D hat mass operator and $K$ the 1D diagonal hat stiffness operator.
The `d`-dimensional Helmholtz Galerkin operator has the tensor-sum structure

```math
A = c\,M^{\otimes d} + \sum_{s=1}^d \Bigl(K^{(s)} \otimes \bigotimes_{j\neq s} M^{(j)}\Bigr),
```

where $K^{(s)}$ acts in the $s$-th dimension and $M^{(j)}$ in all remaining ones.

In the code this is represented as a `TensorSumMatVec` built from `WeightedTensorTerm`s.
Each tensor term is evaluated by an `UpDownTensorOp`, which orchestrates the necessary 1D sweeps
(and uses `updown(op)` when the 1D operator needs splitting).

## Right-hand side used in the example

To keep the example compact and reproducible, we use the same "single-mode" forcing idea as in the
paper’s numerical section.

We pick a tensor-product hierarchical hat function

```math
w(x) = \bigotimes_{j=1}^d t(x_j),
```

where `t` is the *level-1 interior hat* function. We set

```math
f(x) = c\,w(x).
```

In a Galerkin method, $b_i = \int_\Omega f\,\varphi_i\,dx$, so for this manufactured forcing the
right-hand side vector is

```math
b = c\,(M^{\otimes d})\,e_w,
```

where $e_w$ is the coefficient vector that selects the single basis function $w$.
Computing $b$ is therefore just an application of the fast mass operator.
