"""Shen & Yu (2010) Section 4 sparse grid Poisson examples.

This script reproduces a small subset of the CH1/CH2 Chebyshev–Legendre–Dirichlet
pipeline used in Shen & Yu (2010), focusing on:

1. sampling `f` on the CH1 sparse grid,
2. transforming CH1 nodal values to Chebyshev coefficients,
3. Chebyshev→Legendre modal transform,
4. applying the Dirichlet RHS contraction,
5. restriction CH1→CH2, and solving the CH2 Galerkin system with CG.

Run:

    julia --project=. examples/shen_yu_2010_sec4.jl --case=3 --d=3 --level=6 --k=3 --alpha=0.0

When included, this file only defines helper functions.
"""

using UnifiedSparseGrids
import UnifiedSparseGrids: lineop_style, apply_line!
using StaticArrays
using LinearAlgebra
using BandedMatrices
using Krylov

# -----------------------------------------------------------------------------
# 1D connection / RHS operators

# Chebyshev→Legendre conversion is applied on coefficient fibers with
# `LineChebyshevLegendre(Val(:forward))`.
#
# The 1D Dirichlet Legendre RHS contraction (padded to preserve fiber length)
# is applied in-place with `LineLegendreDirichletRHS()`.

"""Dirichlet RHS contraction in the Legendre basis, padded to preserve length.

Given Legendre coefficients `c[1:N+1]` (degrees `0:N`), this maps in-place to a
padded RHS vector `bp` with `bp[1:2]=0` and, for `i=1:N-1`,

    bp[i+2] = 2/(2i-1)*c[i] - 2/(2i+3)*c[i+2].

This is a convenience line operator used only in this example script.
"""
struct LineLegendreDirichletRHS <: AbstractLineOp end

lineop_style(::Type{LineLegendreDirichletRHS}) = InPlaceOp()

@inline function apply_line!(::LineLegendreDirichletRHS, buf::AbstractVector{T}, work::AbstractVector{T},
                             ::AbstractUnivariateNodes,
                             ::Int, plan) where {T<:Number}
    n = length(buf)
    n >= 2 || throw(DimensionMismatch("expected length ≥ 2, got $n"))
    m = n - 2
    @inbounds for i in m:-1:1
        buf[i + 2] = (T(2) / T(2i - 1)) * buf[i] - (T(2) / T(2i + 3)) * buf[i + 2]
    end
    @inbounds if n >= 1
        buf[1] = zero(T)
        n >= 2 && (buf[2] = zero(T))
    end
    return buf
end

# -----------------------------------------------------------------------------
# High-level RHS assembly helper

"""Assemble the Galerkin RHS in the Dirichlet Legendre basis (CH2 grid)."""
function assemble_dirichlet_rhs!(rhsI::AbstractVector, fJ::AbstractVector,
                                 gridI::SparseGrid{<:SparseGridSpec{D}},
                                 gridJ::SparseGrid{<:SparseGridSpec{D}};
                                 planJ=nothing, tplan::Union{Nothing,TransferPlan}=nothing) where {D}
    length(fJ) == length(gridJ) || throw(DimensionMismatch("input length mismatch"))
    length(rhsI) == length(gridI) || throw(DimensionMismatch("output length mismatch"))

    planJ === nothing && (planJ = CyclicLayoutPlan(gridJ, eltype(fJ)))

    axesJ = gridJ.spec.axes

    opA = tensorize(CompositeLineOp(LineTransform(Val(:forward)), LineHierarchize()), Val(D))
    opB = tensorize(CompositeLineOp(LineDehierarchize(),
                                    LineChebyshevLegendre(Val(:forward)),
                                    LineLegendreDirichletRHS()), Val(D))
    chain = compose(opA, opB)

    u = OrientedCoeffs{D}(copy(fJ))
    apply_unidirectional!(u, gridJ, chain, planJ)

    if tplan === nothing
        axesI = gridI.spec.axes
        shift = SVector{D,Int}(ntuple(d -> totalsize(axesJ[d], 0) - totalsize(axesI[d], 0), Val(D)))
        coord_map = coordsI -> coordsI + shift
        tplan = TransferPlan(gridI, gridJ; coord_map=coord_map)
    end
    restrict!(rhsI, u.data, tplan)
    return rhsI
end

# -----------------------------------------------------------------------------
# Dirichlet Legendre mass/stiffness operators (CH2) + matrix-free matvec

"""1D mass matrix in the Dirichlet Legendre basis ϕₖ = Lₖ − Lₖ₊₂.

This is a matrix family function compatible with `LineBandedOp`:

    A = legendre_dirichlet_mass_matrix(n, T)
"""
function legendre_dirichlet_mass_matrix(n::Integer, ::Type{T}) where {T<:Number}
    n = Int(n)
    n <= 0 && return BandedMatrices.BandedMatrix{T}(undef, (0, 0), (0, 0))

    A = BandedMatrices.BandedMatrix{T}(undef, (n, n), (2, 2))
    fill!(A, zero(T))

    @inbounds for i in 1:n
        k = i - 1
        A[i, i] = T(2) / T(2k + 1) + T(2) / T(2k + 5)
        if i + 2 <= n
            v = -T(2) / T(2k + 5)
            A[i, i + 2] = v
            A[i + 2, i] = v
        end
    end
    return A
end

"""1D stiffness diagonal in the Dirichlet Legendre basis for ∫ u' v'.

This is a diagonal family function compatible with `LineDiagonalOp`:

    d = legendre_dirichlet_stiffness_diag(n, T)
"""
function legendre_dirichlet_stiffness_diag(n::Integer, ::Type{T}) where {T<:Number}
    n = Int(n)
    n <= 0 && return T[]
    d = Vector{T}(undef, n)
    @inbounds for i in 1:n
        k = i - 1
        d[i] = T(2) * T(2k + 3)
    end
    return d
end

"""Diagonal of the dD mass operator on a Dirichlet (CH2) grid."""
function dirichlet_mass_diag(gridI::SparseGrid{<:SparseGridSpec{D}};
                             T::Type{<:Number}=Float64) where {D}
    any(ax -> !(ax isa ChebyshevGaussLobattoNodes && totalsize(ax, 0) == 0), gridI.spec.axes) &&
        throw(ArgumentError("dirichlet_mass_diag expects ChebyshevGaussLobattoNodes(endpoints=false) in every dimension"))

    diag = Vector{T}(undef, length(gridI))
    it = traverse(gridI)
    @inbounds for (i, c) in enumerate(it)
        massprod = one(T)
        for j in 1:D
            k = Int(c[j]) - 1
            massprod *= T(2) / T(2k + 1) + T(2) / T(2k + 5)
        end
        diag[i] = massprod
    end
    return diag
end

"""Diagonal of the dD stiffness operator on a Dirichlet (CH2) grid."""
function dirichlet_stiff_diag(gridI::SparseGrid{<:SparseGridSpec{D}};
                              T::Type{<:Number}=Float64) where {D}
    any(ax -> !(ax isa ChebyshevGaussLobattoNodes && totalsize(ax, 0) == 0), gridI.spec.axes) &&
        throw(ArgumentError("dirichlet_stiff_diag expects ChebyshevGaussLobattoNodes(endpoints=false) in every dimension"))

    diag = Vector{T}(undef, length(gridI))
    it = traverse(gridI)
    @inbounds for (i, c) in enumerate(it)
        md = ntuple(j -> begin
            k = Int(c[j]) - 1
            T(2) / T(2k + 1) + T(2) / T(2k + 5)
        end, Val(D))
        massprod = one(T)
        for j in 1:D
            massprod *= md[j]
        end
        stiff = zero(T)
        for s in 1:D
            ks = Int(c[s]) - 1
            stiff += (T(2) * T(2ks + 3)) * (massprod / md[s])
        end
        diag[i] = stiff
    end
    return diag
end

"""Build the sparse grid Galerkin mass operator `M` on a Dirichlet (CH2) grid."""
function dirichlet_mass_operator(gridI::SparseGrid{<:SparseGridSpec{D}};
                                 Ti::Type{<:Integer}=Int,
                                 T::Type{<:Number}=Float64) where {D}
    any(ax -> !(ax isa ChebyshevGaussLobattoNodes && totalsize(ax, 0) == 0), gridI.spec.axes) &&
        throw(ArgumentError("dirichlet_mass_operator expects ChebyshevGaussLobattoNodes(endpoints=false) in every dimension"))

    mass1d = LineBandedOp(legendre_dirichlet_mass_matrix)
    mass_ops = ntuple(_ -> mass1d, Val(D))
    term = WeightedTensorTerm(one(T), UpDownTensorOp(Tuple(mass_ops); omit_dim=0))
    return TensorSumMatVec(gridI, (term,), T; Ti=Ti)
end

"""Build the sparse grid Galerkin stiffness operator `S` on a Dirichlet (CH2) grid."""
function dirichlet_stiffness_operator(gridI::SparseGrid{<:SparseGridSpec{D}};
                                      Ti::Type{<:Integer}=Int,
                                      T::Type{<:Number}=Float64) where {D}
    any(ax -> !(ax isa ChebyshevGaussLobattoNodes && totalsize(ax, 0) == 0), gridI.spec.axes) &&
        throw(ArgumentError("dirichlet_stiffness_operator expects ChebyshevGaussLobattoNodes(endpoints=false) in every dimension"))

    mass1d = LineBandedOp(legendre_dirichlet_mass_matrix)
    stiff1d = LineDiagonalOp(legendre_dirichlet_stiffness_diag)
    mass_ops = ntuple(_ -> mass1d, Val(D))
    stiff_ops = ntuple(_ -> stiff1d, Val(D))

    terms = ntuple(s -> begin
        ops_s = ntuple(d -> (d == s ? stiff_ops[d] : mass_ops[d]), Val(D))
        WeightedTensorTerm(one(T), UpDownTensorOp(Tuple(ops_s); omit_dim=0))
    end, Val(D))

    return TensorSumMatVec(gridI, terms, T; Ti=Ti)
end


"""Solve `(α*M + S)u = b` on a CH2 grid with a matrix-free Krylov method."""
function solve_dirichlet_mass_stiffness(gridI::SparseGrid{<:SparseGridSpec{D}},
                                        b::AbstractVector{T};
                                        α::Real=0,
                                        precond::Symbol=:jacobi,
                                        rtol::Real=sqrt(eps(T)),
                                        atol::Real=sqrt(eps(T)),
                                        itmax::Int=0,
                                        verbose::Int=0,
                                        Ti::Type{<:Integer}=Int) where {D,T<:Real}
    length(b) == length(gridI) || throw(DimensionMismatch("b has length $(length(b)), expected $(length(gridI))"))

    αT = T(α)
    Mop = dirichlet_mass_operator(gridI; Ti=Ti, T=T)
    Sop = dirichlet_stiffness_operator(gridI; Ti=Ti, T=T)

    M = as_linear_operator(Mop; symmetric=true, hermitian=true)
    S = as_linear_operator(Sop; symmetric=true, hermitian=true)
    Aop = S + αT * M

    P = if precond === :none
        I
    elseif precond === :jacobi
        diagM = dirichlet_mass_diag(gridI; T=T)
        diagS = dirichlet_stiff_diag(gridI; T=T)
        jacobi_precond(diagS .+ αT .* diagM)
    else
        throw(ArgumentError("unsupported precond=$precond (use :jacobi or :none)"))
    end

    x, stats = Krylov.cg(Aop, b; M=P, ldiv=false, rtol=T(rtol), atol=T(atol), itmax=itmax, verbose=verbose)
    return x, stats
end

# -----------------------------------------------------------------------------
# Shen & Yu manufactured solutions


@inline function u1(x::SVector{D,T}; k::Int=1) where {D,T}
    c = T(k) / T(2)
    v = one(T)
    @inbounds for j in 1:D
        v *= sinpi(c * (x[j] + one(T)))
    end
    return v
end

@inline function f1(x::SVector{D,T}; α::T, k::Int=1) where {D,T}
    u = u1(x; k=k)
    return α * u + T(D) * (T(k)^2 * T(pi)^2 / T(4)) * u
end

@inline function φk(x::T, k::Int) where {T}
    s = T(k) * T(pi) * (x + one(T)) / T(2)
    return exp(sin(s)) - one(T)
end

@inline function φk_dd(x::T, k::Int) where {T}
    c = T(k) * T(pi) / T(2)
    s = c * (x + one(T))
    return c^2 * exp(sin(s)) * (cos(s)^2 - sin(s))
end

@inline function u2(x::SVector{D,T}; k::Int=2) where {D,T}
    v = zero(T)
    @inbounds for i in 1:D
        t = φk(x[i], k)
        @inbounds for j in 1:D
            j == i && continue
            t *= sinpi((x[j] + one(T)) / T(2))
        end
        v += t
    end
    return v
end

@inline function f2(x::SVector{D,T}; α::T, k::Int=2) where {D,T}
    u = u2(x; k=k)
    acc = α * u
    @inbounds for i in 1:D
        t = ((T(D - 1) * T(pi)^2) / T(4)) * φk(x[i], k) - φk_dd(x[i], k)
        @inbounds for j in 1:D
            j == i && continue
            t *= sinpi((x[j] + one(T)) / T(2))
        end
        acc += t
    end
    return acc
end

@inline function gk(x::T, k::Int) where {T}
    ϵ = T(10.0)^(-T(k))
    p = one(T) + x - x^2 - x^3
    return p * log(one(T) + x + ϵ)
end

@inline function gk_dd(x::T, k::Int) where {T}
    ϵ = T(10.0)^(-T(k))
    p = one(T) + x - x^2 - x^3
    p1 = one(T) - T(2) * x - T(3) * x^2
    p2 = -T(2) - T(6) * x
    q = one(T) + x + ϵ
    l = log(q)
    l1 = inv(q)
    l2 = -inv(q^2)
    return p2 * l + T(2) * p1 * l1 + p * l2
end

@inline function u3(x::SVector{D,T}; k::Int=3) where {D,T}
    v = one(T)
    @inbounds for j in 1:D
        v *= gk(x[j], k)
    end
    return v
end

@inline function f3(x::SVector{D,T}; α::T, k::Int=3) where {D,T}
    g = ntuple(j -> gk(x[j], k), Val(D))
    u = one(T)
    @inbounds for j in 1:D
        u *= g[j]
    end

    acc = α * u
    @inbounds for i in 1:D
        prod_other = one(T)
        @inbounds for j in 1:D
            j == i && continue
            prod_other *= g[j]
        end
        acc += (-gk_dd(x[i], k)) * prod_other
    end
    return acc
end

@inline hk(x::T, k::Int) where {T} = (x <= zero(T) ? zero(T) : x^k)

@inline function u4(x::SVector{D,T}; k::Int=3) where {D,T}
    v = one(T)
    @inbounds for j in 1:D
        v *= (hk(x[j], k) - (one(T) + x[j]) / T(2))
    end
    return v
end

@inline function f4(x::SVector{D,T}; α::T, k::Int=3) where {D,T}
    a = ntuple(j -> (hk(x[j], k) - (one(T) + x[j]) / T(2)), Val(D))
    u = one(T)
    @inbounds for j in 1:D
        u *= a[j]
    end

    acc = α * u
    @inbounds for i in 1:D
        prod_other = one(T)
        @inbounds for j in 1:D
            j == i && continue
            prod_other *= a[j]
        end
        acc += (T(k) * T(k - 1) * hk(x[i], k - 2)) * prod_other
    end
    return acc
end


function run_case(; case::Int=1, d::Int=3, level::Int=5, k::Int=1, α::Float64=0.0)
    cap = fill(level, d)
    I = SmolyakIndexSet(d, level; cap=cap)

    # NOTE: Sparse-grid traversal and the unidirectional transform engine assume
    # nested 1D data are stored in the package's hierarchical Δ-level order.
    # For Chebyshev CGL / Dirichlet grids this corresponds to `LevelOrder()`.
    nodesJ = ntuple(_ -> ChebyshevGaussLobattoNodes(LevelOrder()), d)
    nodesI = ntuple(_ -> ChebyshevGaussLobattoNodes(LevelOrder(); endpoints=false), d)
    gridJ = SparseGrid(SparseGridSpec(nodesJ, I))
    gridI = SparseGrid(SparseGridSpec(nodesI, I))

    f = case == 1 ? (x -> f1(x; α=α, k=k)) :
        case == 2 ? (x -> f2(x; α=α, k=k)) :
        case == 3 ? (x -> f3(x; α=α, k=k)) :
        case == 4 ? (x -> f4(x; α=α, k=k)) :
        throw(ArgumentError("case must be 1..4"))

    fJ = evaluate(gridJ, f)

    bI = zeros(Float64, length(gridI))
    assemble_dirichlet_rhs!(bI, fJ, gridI, gridJ)

    u, stats = solve_dirichlet_mass_stiffness(gridI, bI; α=α, precond=:jacobi, rtol=1e-10, atol=1e-12, itmax=2000)

    # --- discrete L2 error on a tensor grid (no quadrature) ---
    uex = case == 1 ? (x -> u1(x; k=k)) :
          case == 2 ? (x -> u2(x; k=k)) :
          case == 3 ? (x -> u3(x; k=k)) :
          case == 4 ? (x -> u4(x; k=k)) :
          error("unreachable")

    L2, relL2 = discrete_l2_error(u, gridI, DirichletLegendreBasis(), uex, level)

    println("case=$case d=$d level=$level k=$k α=$α")
    println("  n = $(length(gridI)) (CH2 unknowns), CG iters = $(stats.niter), solved=$(stats.solved)")
    println("  discrete L2 error: $(L2)")
    println("  rel discrete L2 error: $(relL2)")
    return (u=u, stats=stats, L2=L2, relL2=relL2)
end

"""Parse `--key=value` style arguments from `ARGS`."""
function _parse_kv_args(args)
    kv = Dict{String,String}()
    for a in args
        startswith(a, "--") || continue
        s = a[3:end]
        i = findfirst(==( '=' ), s)
        i === nothing && continue
        kv[s[1:i-1]] = s[i+1:end]
    end
    return kv
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = _parse_kv_args(ARGS)
    case  = parse(Int, get(kv, "case",  "1"))
    d     = parse(Int, get(kv, "d",     "2"))
    level = parse(Int, get(kv, "level", "6"))
    k     = parse(Int, get(kv, "k",     "3"))
    α     = parse(Float64, get(kv, "alpha", "0.0"))
    run_case(; case, d, level, k, α)
end
