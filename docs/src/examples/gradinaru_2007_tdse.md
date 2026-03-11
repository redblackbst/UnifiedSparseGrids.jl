# Time-dependent Schrödinger equation on sparse grids (Gradinaru 2007)

This example implements the numerical setup in Section 4 of:

- V. Gradinaru (2007), *Fourier transform on sparse grids: Code design and the time dependent Schrödinger equation*.

The corresponding implementation is in `examples/gradinaru_2007_tdse.jl`.

## Model problem

We consider the periodic time-dependent Schrödinger equation (TDSE) on

```math
\Omega = [0,2\pi]^d
```

```math
i\,\varepsilon\,\partial_t u
= -\frac{\varepsilon^2}{2}\,\Delta u + V(x)\,u,
\qquad u(x,0)=g(x),
```

with the harmonic potential

```math
V(x) = \frac12 \sum_{s=1}^d (x_s - \pi)^2.
```

Gradinaru uses Gaussian initial data (the example script follows the same choice).

## Spatial discretization

### Sparse grid Fourier collocation

The spatial discretization is Fourier collocation on a sparse grid:

- Each 1D rule is `FourierEquispacedNodes()`.
- The multi-index set is `SmolyakIndexSet(d, n)` (paper resolution $n$).

**Why `LevelOrder()` for the nodal data?**

The recursive sparse grid layout and the unidirectional transform engine require that
refining the 1D rule appends new points *after* the existing ones (nested-prefix order). For Fourier
nodes this is provided by `LevelOrder()` (bit-reversal ordering on equispaced grids).

Internally, the example evaluates $V$ and $g$ on the sparse grid in the package’s recursive layout
(order consistent with `traverse(grid)`), and stores the evolving wave function $u$ as a single
complex vector.

## Time discretization: Strang splitting

Following Gradinaru, we apply Strang splitting over a time step $\tau$.
In the "physical-space" formulation (paper Eq. (11)):

```math
u \leftarrow
\exp\Bigl(-i\,\tfrac{\tau}{2\varepsilon}V\Bigr)
\,F^{-1}
\,\exp\Bigl(-i\,\tfrac{\tau\varepsilon}{2}|\omega|^2\Bigr)
\,F
\,\exp\Bigl(-i\,\tfrac{\tau}{2\varepsilon}V\Bigr)
\,u,
```

where $F$ / $F^{-1}$ are the **sparse grid Fourier transforms**.

In code:

- `ForwardTransform(d)` implements $F$.
- `InverseTransform(d)` implements $F^{-1}$.

Both are matrix-free and evaluated by `apply_unidirectional!(...)` (fiber-by-fiber sweeps).

### Diagonal phases

The Strang step becomes cheap once we precompute the diagonal phases:

- `phaseV = exp(-i * (τ/(2ε)) * V(x))` (pointwise multiplication in physical space).
- `phaseK = exp(-i * (τ*ε/2) * |ω|^2)` (pointwise multiplication in Fourier space).

This is exactly what the example does.

## Frequency indexing: σ-order in the paper vs natural order here

Gradinaru’s paper reindexes Fourier modes using the *zig-zag* map $\sigma$ (their Section 2),
so that a single index $q = 0,\dots,2^\ell-1$ maps to a signed frequency $\sigma(q)$:

```math
\sigma(q) =
\begin{cases}
-q/2, & q \text{ even},\\
(q+1)/2, & q \text{ odd}.
\end{cases}
```

This "σ-order" is convenient for exposition because it enumerates the symmetric frequency set
$\{1-2^{\ell-1},\dots,2^{\ell-1}\}$ in a nested way.

### What we do instead

`UnifiedSparseGrids` intentionally keeps the output of the 1D FFT in standard FFTW **natural order**
(indices $k = 0,\dots,N-1$). This avoids extra global permutations to/from σ-order, and fits cleanly
into the in-place, fiber-by-fiber **unidirectional** transform engine.

The key invariant required by the unidirectional engine is not the σ-order per se, but the **nested-prefix property** of level-dependent fibers:

- for a nested 1D rule, the data for level $\ell$ occupies the first `npoints(nodes, ℓ)` entries of the
  level $\ell+1$ fiber.

This is why `LevelOrder()` is used for nodal data, and why we avoid introducing additional
reorderings in the modal representation.

### Wrap-around map for physical frequencies

Because modal coefficients are stored in FFTW natural order, the mapping from array index
$k \in \{0,\dots,N-1\}$ to the signed Fourier frequency $\omega \in \mathbb{Z}$ is the usual wrap-around:

```math
\omega(k;N) =
\begin{cases}
\phantom{-}k, & 0 \le k \le N/2,\\
\phantom{-}k-N, & N/2 < k < N.
\end{cases}
```

In $d$ dimensions we use $|\omega|^2 = \sum_{j=1}^d \omega_j^2$.

On sparse grids, a coefficient belongs to some tensor subspace $W_{\ell}$.
Therefore the signed frequency must be interpreted with the corresponding *local* 1D resolution
$N_j = npoints(\mathrm{nodes}[j], \ell_j)$.
In the implementation we infer $\ell_j$ from the nested-prefix coordinate convention:

```math
\ell_j = \min\{\ell \ge 0 : c_j \le npoints(\mathrm{nodes}[j], \ell)\},
\qquad
N_j = npoints(\mathrm{nodes}[j], \ell_j),
\qquad
k_j = c_j - 1.
```

This is the only place where the "ordering difference" matters: the kinetic phase depends on the
*physical* frequency, and with natural order we must apply the wrap-around map.
