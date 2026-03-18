# Sparse spectral Galerkin elliptic solve (Shen–Yu 2010)

This example follows the construction in:

- J. Shen and H. Yu (2010), *Efficient Spectral Sparse Grid Methods and Applications to High-Dimensional Elliptic Problems*.

It is implemented in `examples/shen_yu_2010_sec4.jl`.

The goal of the example is twofold:

1. Demonstrate how to assemble the **right-hand side** for a sparse spectral Galerkin method
   efficiently.
2. Solve the resulting system using a **matrix-free** sparse grid operator evaluated via the
   unidirectional engine.

## Model problem

Shen–Yu consider the Dirichlet elliptic model problem (paper Eq. (3.1)) on

```math
\Omega = (-1,1)^d
```

```math
\begin{aligned}
\alpha u - \Delta u &= f && \text{in }\Omega,\\
 u &= 0 && \text{on }\partial\Omega,
\end{aligned}
```

with $\alpha \ge 0$.

## CH1 vs CH2 grids and spaces

A core idea in Shen–Yu is to decouple:

- the grid used for *sampling/interpolating* `f` (**CH1**), and
- the approximation space used for the Galerkin solution (**CH2**).

In our code:

- **CH1 sparse grid** `gridJ` uses `ChebyshevGaussLobattoNodes()`.
- **CH2 sparse grid** `gridI` uses `ChebyshevGaussLobattoNodes(; endpoints=false)`.

Both use the same Smolyak index set `SmolyakIndexSet(d, level; cap=...)` for simplicity.

`LevelOrder()` is essential: it stores each refinement as an append-only extension, which is the
ordering assumption behind the recursive layout and the unidirectional sweeps.

## Galerkin formulation and the right-hand side

Let $V_d^q$ be the sparse approximation space (CH2) spanned by a **Dirichlet Legendre basis**

```math
\varphi_k(x) = L_k(x) - L_{k+2}(x),
```

and let $U_d^q$ be the sparse interpolation operator (CH1).

Shen–Yu’s sparse spectral Galerkin method (paper Eq. (3.6)) is:

```math
\text{Find } u_d^q \in V_d^q \text{ such that }
\alpha(u_d^q,v) - (\Delta u_d^q,v) = (U_d^q f, v),
\quad \forall v\in V_d^q.
```

This produces a linear system

```math
(\alpha M + S)\,u = b,
```

with mass matrix $M$, stiffness matrix $S$, and right-hand side coefficients

```math
b_k = (U_d^q f, \varphi_k).
```

### How the example computes $b$ efficiently

Section 3.2 of Shen–Yu describes how to compute $b_k$ without ever forming dense multi-dimensional
transform matrices.

In our implementation, `assemble_dirichlet_rhs!(bI, fJ, gridI, gridJ)` performs the following chain:

1. **Sample $f$ on CH1**:
   - `fJ = evaluate(gridJ, f)` returns values ordered consistently with `traverse(gridJ)`.

2. **Compute hierarchical Chebyshev coefficients** ($\tilde f_k$ in the paper):

   - Apply 1D Chebyshev transforms (DCT-I) and hierarchize on each fiber.

   In code this is

   ```julia
   opA = tensorize(CompositeLineOp(LineTransform(Val(:forward)), LineHierarchize()), Val(D))
   ```

3. **Convert to standard Chebyshev, then to Legendre** (paper steps 2–3):

   - `LineDehierarchize()` rewrites hierarchical Chebyshev coefficients into standard Chebyshev
     coefficients.
   - `LineChebyshevLegendre(Val(:forward))` applies the (triangular) Chebyshev→Legendre conversion.

4. **Contract with the Dirichlet basis** to form $(U_d^q f, \varphi_k)$ (paper step 4):

   - This is implemented by `LineLegendreDirichletRHS`, a custom `AbstractLineOp` defined in the
     example file.

   In code this second sweep is

   ```julia
   opB = tensorize(
       CompositeLineOp(LineDehierarchize(), LineChebyshevLegendre(Val(:forward)), LineLegendreDirichletRHS()),
       Val(D),
   )
   ```

5. **Compose two full sweeps** and apply them in a single call:

   ```julia
   chain = compose(opA, opB)
   apply_unidirectional!(u, gridJ, chain, planJ)
   ```

6. **Restrict from CH1 (gridJ) to CH2 (gridI)**:

   The Galerkin unknowns live on CH2. We transfer the assembled RHS coefficients using a
   `TransferPlan` and the exported convenience

   - `restrict!(bI, rhsJ, plan)`.

`LineLegendreDirichletRHS` is intentionally included as an example of how to extend the package’s
**line operator interface** (`AbstractLineOp`, `lineop_style`, `apply_line!`) so that new 1D building
blocks can be composed into higher-dimensional tensor operators.

## Solving: matrix-free (αM + S) on a sparse grid

Once $b$ is available on `gridI`, we solve

```math
(\alpha M + S)\,u = b
```

using CG with a simple Jacobi preconditioner.

### Tensor-sum structure

Just as in the paper, the operator can be written as a sum of tensor-product terms

```math
\alpha\,M^{\otimes d} + \sum_{s=1}^d \Bigl(S^{(s)} \otimes \bigotimes_{j\neq s} M^{(j)}\Bigr).
```

In the Dirichlet Legendre basis used here:

- the 1D stiffness operator is **diagonal**, and
- the 1D mass operator is **banded**.

These are implemented by

- `LineDiagonalOp` (stiffness), whose family function returns the diagonal
  entries as a vector, and
- `LineBandedOp` (mass), whose family function returns a `BandedMatrix`.

The full $d$-dimensional operator is represented as a `TensorSumMatVec` of
`WeightedTensorTerm`s, and applied without ever forming $M$ or $S$ explicitly.
