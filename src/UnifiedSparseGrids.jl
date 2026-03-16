module UnifiedSparseGrids

using StaticArrays
using LinearAlgebra
using BandedMatrices
using ClassicalOrthogonalPolynomials
using LinearOperators
using FastTransforms
using DataStructures
using Base.Threads

export AbstractMeasure,
       LegendreMeasure, ChebyshevTMeasure, ChebyshevUMeasure,
       JacobiMeasure, LaguerreMeasure, HermiteMeasure,
       FourierMeasure, UnitIntervalMeasure

export AbstractNodeOrder, NaturalOrder, LevelOrder,
       bitrev, bitrevperm

export AbstractAxisFamily, AbstractNestedAxisFamily, AbstractNonNestedAxisFamily,
       AbstractUnivariateNodes, AbstractNestedNodes, AbstractNonNestedNodes,
       GaussNodes, GaussLobattoNodes, PattersonNodes,
       ChebyshevGaussLobattoNodes, ClenshawCurtisNodes,
       EquispacedNodes, FourierEquispacedNodes, DyadicNodes,
       LejaNodes, PseudoGaussNodes,
       measure, nodeorder, is_nested,
       totalsize, blocksize, refinement_index,
       npoints, delta_count, points, newpoints

export AbstractIndexSet, SmolyakIndexSet, WeightedSmolyakIndexSet,
       FullTensorIndexSet,
       dim, refinement_caps

export SparseGridSpec, SparseGrid,
       refinement_caps

# Cross-grid / evaluation point sets
export CrossGridPlan, TransferPlan, subgrid_index_map,
       restrict!, embed!,
       AbstractPointSet, ScatteredPoints, TensorProductPoints,
       EvaluationPlan, plan_evaluate, evaluate!, evaluate, SparseEvalOp,
       AbstractUnivariateBasis,
       ChebyshevTBasis, ChebyshevUBasis, LegendreBasis, DirichletLegendreBasis, FourierBasis,
       HatBasis, PiecewisePolynomialBasis,
       support_style, active_local, eval_local,
       ncoeff, vandermonde

export AbstractLayout, RecursiveLayout, SubspaceLayout,
       recursive_to_subspace, subspace_to_recursive,
       recursive_to_subspace!, subspace_to_recursive!,
       SubspaceBlock, each_subspace_block,
       traverse,
       plot_subspace_layout, plot_combination_technique,
       plot_sparse_grid, plot_sparse_indexset

export CombinationSubproblem, each_combination_subproblem

# Adaptive quadrature
export AbstractQuadratureFamily, AbstractNestedQuadratureFamily,
       AbstractQuadraturePointFamily, AbstractNestedQuadraturePointFamily,
       QuadratureRule,
       ClenshawCurtisQuadrature, GaussLegendreQuadrature, GaussLaguerreQuadrature, GaussHermiteQuadrature,
       WeightedLejaPoints, WeightedLejaQuadrature, PseudoGaussQuadrature,
       MappedGaussianQuadrature, IdentityMap, ReciprocalExpMap,
       QuadratureFamily,
       qrule, qdiffrule, qdegree, qmeasure,
       qsize, qpoints, qweights,
       AdaptiveQuadratureState,
       delta_contribution,
       integrate_adaptive

# 1D basis-change kernels
export dirichlet_to_legendre!, legendre_to_dirichlet!, legendre_dirichlet_rhs!

# Sparse grid Galerkin utilities (general)
export WeightedTensorTerm, TensorSumMatVec,
       as_linear_operator, jacobi_precond,
       discrete_l2_error,
       HatMassLineOp, HatStiffnessDiagLineOp

# Unidirectional sparse grid application
export OrientedCoeffs, cycle_last_to_front,
       each_lastdim_fiber,
       CyclicLayoutPlan,
       AbstractLineOp,
       LineOpStyle, InPlaceOp, OutOfPlaceOp,
       lineop_style,
       needs_plan, lineplan,
       apply_line!,
       IdentityLineOp, ZeroLineOp, LineTransform, LineChebyshevLegendre, LineHierarchize, LineDehierarchize, LineHierarchizeTranspose, LineDehierarchizeTranspose,
       LineMatrixOp, LineDiagonalOp, LineBandedOp, UpDownTensorOp,
       CompositeLineOp, lineops,
       AbstractTensorOp, TensorOp, CompositeTensorOp, lineop, tensorize,
       ForwardTransform, InverseTransform,
       compose,
       apply_lastdim_cycled!, apply_unidirectional!

include("measures.jl")
include("orders.jl")
include("nodes.jl")
include("basis.jl")
include("indexsets.jl")
include("layout.jl")
include("combination.jl")
include("operators.jl")
include("transforms.jl")
include("unidirectional.jl")
include("crossgrid.jl")
include("evaluation.jl")
include("quadrature.jl")
include("galerkin.jl")

end # module UnifiedSparseGrids
