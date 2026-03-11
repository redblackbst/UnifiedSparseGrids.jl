# Adaptive quadrature

`UnifiedSparseGrids.jl` provides a dimension-adaptive sparse-grid quadrature subsystem
for genuine scalar callback integrals. The implementation follows the generalized
sparse-grid viewpoint of Gerstner and Griebel (2003, *Computing* 71(1):65-87,
[doi:10.1007/s00607-003-0015-5](http://link.springer.com/10.1007/s00607-003-0015-5))
and the dimension-adaptive sparse-grid construction summarized in the chapter
"Dimension-Adaptive Sparse Grid Quadrature" in Garcke (ed., 2014),
*Sparse Grids and Applications - Munich 2012*
([doi:10.1007/978-3-319-04537-5](http://link.springer.com/10.1007/978-3-319-04537-5)).

## Generalized sparse-grid quadrature

Let

```math
I(f) = \int_\Omega f(x)\,dx, \qquad \Omega = \Omega_1 \times \cdots \times \Omega_D,
```

and let $Q_r^{(d)}$ denote the one-dimensional quadrature rule of refinement index
$r$ in dimension $d$. Following Gerstner--Griebel, the hierarchical difference rules are

```math
\Delta_0^{(d)} = Q_0^{(d)}, \qquad
\Delta_r^{(d)} = Q_r^{(d)} - Q_{r-1}^{(d)} \quad (r \ge 1).
```

For a multi-index $r = (r_1, \ldots, r_D)$, the tensor-difference contribution is

```math
\Delta_r f = (\Delta_{r_1}^{(1)} \otimes \cdots \otimes \Delta_{r_D}^{(D)}) f.
```

Given an admissible index set $I \subset \mathbb{N}_0^D$, the generalized sparse-grid
quadrature formula is

```math
Q_I f = \sum_{r \in I} \Delta_r f.
```

Admissibility means downward closure:

```math
r \in I, \; r_j > 0 \implies r - e_j \in I \qquad (j = 1, \ldots, D).
```

This condition guarantees that the telescoping representation is valid. In the papers
above the root index is written as $(1,\ldots,1)$ because the univariate rules are
numbered starting from one. `UnifiedSparseGrids.jl` uses zero-based refinement indices,
so the root contribution is $r = (0,\ldots,0)$ and the same formulas apply after this
index shift.

## Dimension-adaptive refinement

The adaptive algorithm maintains two subsets of the current admissible index set:

- an **active** frontier, whose tensor-difference values have been computed but whose
  forward neighbours have not all been explored yet;
- an **old** set, whose forward neighbours have already been considered.

At each step the algorithm removes one active index, inserts all newly admissible
forward neighbours, and updates

```math
Q_I f = \sum_{r \in I} \Delta_r f, \qquad
\eta = \sum_{r \in A} |\Delta_r f|,
```

where $A$ is the active set. This is the same active/old bookkeeping pattern used in the
Gerstner--Griebel data structures and in the dimension-adaptive chapter of Garcke (2014),
except that this package stores the state directly as Julia containers (`BinaryMaxHeap`
plus dictionaries for active/old status and pending predecessor counts).

The current implementation exposes three selection priorities:

```math
\texttt{:absdelta}: \quad \pi(r) = |\Delta_r f|,
```

```math
\texttt{:profit}: \quad \pi(r) = \frac{|\Delta_r f|}{\max(\operatorname{work}(r), 1)},
```

```math
\texttt{:normalized}: \quad \pi(r) = \frac{|\Delta_r f|}{\max(\operatorname{nevals}(r), 1)}.
```

The `:profit` mode mirrors the work-aware priorities discussed in Gerstner--Griebel:
it prefers indices that contribute a lot per tensor block cost instead of simply taking the
largest absolute difference.

## 1D quadrature families

The main quadrature entry points are:

- `QuadratureRule`
- `qrule(Q, r)`
- `qdiffrule(Q, r)`
- `qsize(Q, r)`
- `qdegree(Q, r)`
- `qmeasure(Q)`

Available families include:

- `ClenshawCurtisQuadrature()`
- `GaussLegendreQuadrature()`
- `GaussLaguerreQuadrature()`
- `GaussHermiteQuadrature()`
- `PseudoGaussQuadrature(...)`
- `WeightedLejaPoints(...)`
- `WeightedLejaQuadrature(...)`
- `MappedGaussianQuadrature(...)`

For nested families, `delta_contribution(...)` evaluates $\Delta_r$ directly from the
one-dimensional difference rules `qdiffrule(...)`. For non-nested families the package
falls back to the standard inclusion-exclusion expansion of $\Delta_r$ into tensor-product
rules.

## Worked adaptive example

The following example integrates a smooth anisotropic function on $[-1,1]^3$ using a
Clenshaw--Curtis family and a weighted Smolyak admissibility envelope.

```julia
using UnifiedSparseGrids
using StaticArrays

Q = ClenshawCurtisQuadrature()
env = WeightedSmolyakIndexSet(3, 8, (1, 1, 2))

f(x) = exp(-(x[1]^2 + x[2]^2)) * cos(x[3])

val, state = integrate_adaptive(
    f,
    (Q, Q, Q),
    env;
    indicator = :profit,
    rtol = 1e-8,
    maxterms = 5_000,
)
```

On the current implementation this produces:

```text
val = 3.754619100598843
typeof(state) = AdaptiveQuadratureState{SVector{3, Int64}, Float64, Float64, Float64, DataStructures.BinaryMaxHeap{UnifiedSparseGrids.FrontierEntry{SVector{3, Int64}, Float64, Float64, Float64}}, DataStructures.RobinDict{SVector{3, Int64}, Bool}, DataStructures.RobinDict{SVector{3, Int64}, UInt16}}
(state.integral, state.eta, state.nevals, state.ncalls, state.work) = (3.754619100598843, 2.35045416666516e-8, 6144, 70, 6144.0)
(state.nactive, state.nold) = (8, 62)
state.accepted = SVector{3, Int64}[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 1], [0, 1, 1], [0, 0, 2]  …  [4, 2, 0], [2, 4, 0], [4, 1, 1], [4, 0, 2], [1, 4, 1], [0, 4, 2], [3, 4, 0], [2, 4, 1], [4, 3, 0], [4, 2, 1]]
state.frontier_pops = SVector{3, Int64}[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [2, 0, 0], [0, 2, 0]  …  [4, 0, 0], [4, 1, 0], [1, 4, 0], [4, 0, 1], [0, 4, 1], [4, 1, 1], [1, 4, 1], [2, 4, 0], [4, 2, 0], [2, 4, 1]]
state.accepted_contrib = [0.5849757247844771, 0.6701021053245511, 0.6701021053245511, 0.3318043607389496, -0.06564820452399943, 0.7676161805275498, -0.06564820452399941, 0.3800889357741378, 0.3800889357741378, -0.005720347916805356  …  2.9410088822840255e-8, 2.9410088841058114e-8, -1.702780670453756e-7, 2.56268914606856e-9, -1.702780669897059e-7, 2.562689144201575e-9, 8.434585278728775e-10, 1.668171057808188e-8, 8.434585319933859e-10, 1.6681710573036553e-8]
```

A few things are worth noticing when you compare the code with the output:

- `val` and `state.integral` agree, because the state always stores the quadrature sum
  over both active and old indices.
- `state.eta` is the remaining active-set estimate only; it is what the stopping rule
  compares against `atol + rtol * max(1, |state.integral|)`.
- `state.accepted` records the activation order of admissible multi-indices, while
  `state.frontier_pops` records the order in which active indices are moved to old.
- `state.work` is based on the actual tensor-block sizes that were evaluated. With
  `indicator = :profit`, this is the denominator used for frontier priority.

For this particular $f$, the exact integral factorizes,

```math
I(f) = \left(\int_{-1}^1 e^{-x^2} \, dx\right)^2 \left(\int_{-1}^1 \cos z \, dz\right)
= \bigl(\sqrt{\pi}\,\operatorname{erf}(1)\bigr)^2 \cdot 2\sin(1),
```

so the adaptive sparse-grid result is easy to sanity-check against a known target value:

```text
I_exact = 3.7546185280582427
|val - I_exact| = 5.725406002907740e-7
|val - I_exact| / |I_exact| = 1.524896859726711e-7
rtol * max(1, |state.integral|) = 3.754619100598843e-8
```

In this run the algorithm stops because `state.eta = 2.35045416666516e-8` is below the
requested relative tolerance threshold. The true quadrature error is still larger than `state.eta`,
which is a useful reminder that `eta` is the active-frontier estimator used for adaptive stopping,
not a rigorous a posteriori error bound.

The anisotropic envelope `WeightedSmolyakIndexSet(3, 8, (1, 1, 2))` also reflects the
fact that the third direction is allowed to refine more slowly than the first two.

## Slow-growth nested families

`PseudoGaussQuadrature` and `WeightedLejaQuadrature` are especially attractive for
adaptive sparse quadrature because they add only one new point per refinement step:

```julia
PG = PseudoGaussQuadrature(LegendreBasis(), LegendreMeasure())
WL = WeightedLejaQuadrature(LegendreBasis(), LegendreMeasure())
CC = ClenshawCurtisQuadrature()

[(r, qsize(PG, r), qsize(WL, r), qsize(CC, r)) for r in 0:5]
```

```text
6-element Vector{Tuple{Int64, Int64, Int64, Int64}}:
 (0, 1, 1, 2)
 (1, 2, 2, 3)
 (2, 3, 3, 5)
 (3, 4, 4, 9)
 (4, 5, 5, 17)
 (5, 6, 6, 33)
```

That slow growth is often more important than the exact choice of univariate rule,
because it keeps each tensor-difference block small while the adaptive frontier expands.
Clenshaw--Curtis, by contrast, doubles the number of interior intervals at each step,
which quickly increases tensor block sizes.

## Practical notes

- `WeightedSmolyakIndexSet` is often a good static admissibility envelope when you know
  an anisotropy pattern in advance but still want adaptive refinement inside that envelope.
- `delta_contribution(...)` returns `(contribution, nevals, work)`, which makes it easy to
  inspect or prototype custom prioritization strategies.
- For non-nested families the inclusion-exclusion fallback is exact but usually more
  expensive than the nested `qdiffrule(...)` path, so nested slow-growth families are often
  the best fit for dimension-adaptive sparse quadrature.
