"""Sparse grid Galerkin utilities.

This file contains two categories of building blocks:

1. Generic, matrix-free tensor-sum operators and Krylov wrappers.
2. Balder–Zenger hat-basis Helmholtz line operators.
"""

# -----------------------------------------------------------------------------
# Generic Galerkin / linear-solver utilities

"""A weighted tensor-product term in a tensor-sum operator."""
struct WeightedTensorTerm{W,Op}
    weight::W
    op::Op
end

raw"""Matrix-free matvec for a sum of weighted tensor-product operators.

The operator applies

```math
y = \sum_t w_t\,(A_t x),
```

where each $A_t$ is an `AbstractTensorOp` applied via the unidirectional engine.
"""
struct TensorSumMatVec{D,ElT,Terms,PlanT}
    grid::SparseGrid
    terms::Terms
    plan::PlanT
    work::Vector{ElT}
end

function TensorSumMatVec(grid::SparseGrid{<:SparseGridSpec{D}}, terms::Terms, ::Type{ElT}=Float64; Ti::Type{<:Integer}=Int) where {D,Terms<:Tuple,ElT}
    plan = CyclicLayoutPlan(grid, ElT; Ti=Ti)
    work = Vector{ElT}(undef, length(grid))
    return TensorSumMatVec{D,ElT,Terms,typeof(plan)}(grid, terms, plan, work)
end

@inline function _mul_add!(y::AbstractVector{ElT}, A::TensorSumMatVec{D,ElT}, x::AbstractVector{ElT}, α::ElT) where {D,ElT}
    @inbounds for term in A.terms
        copyto!(A.work, x)
        u = OrientedCoeffs{D}(A.work)
        apply_unidirectional!(u, A.grid, term.op, A.plan)
        w = α * convert(ElT, term.weight)
        for i in eachindex(y)
            y[i] += w * A.work[i]
        end
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{ElT}, A::TensorSumMatVec{D,ElT}, x::AbstractVector{ElT}) where {D,ElT}
    length(y) == length(A.grid) || throw(DimensionMismatch("destination length mismatch"))
    length(x) == length(A.grid) || throw(DimensionMismatch("source length mismatch"))
    fill!(y, zero(ElT))
    return _mul_add!(y, A, x, one(ElT))
end

function LinearAlgebra.mul!(y::AbstractVector{ElT}, A::TensorSumMatVec{D,ElT}, x::AbstractVector{ElT}, α, β) where {D,ElT}
    length(y) == length(A.grid) || throw(DimensionMismatch("destination length mismatch"))
    length(x) == length(A.grid) || throw(DimensionMismatch("source length mismatch"))
    αT = convert(ElT, α)
    βT = convert(ElT, β)
    if βT == zero(ElT)
        fill!(y, zero(ElT))
    else
        @inbounds for i in eachindex(y)
            y[i] *= βT
        end
    end
    return _mul_add!(y, A, x, αT)
end

"""Wrap a `TensorSumMatVec` as a `LinearOperators.LinearOperator`.

The returned operator uses the 5-argument `mul!` protocol:

    y ← α*A*x + β*y.
"""
function as_linear_operator(A::TensorSumMatVec{D,ElT}; symmetric::Bool=(ElT <: Real), hermitian::Bool=true) where {D,ElT}
    n = length(A.grid)
    prod! = (res, v, α, β) -> mul!(res, A, v, α, β)
    return LinearOperators.LinearOperator(ElT, n, n, symmetric, hermitian, prod!, prod!, prod!)
end

"""Construct a Jacobi (diagonal) preconditioner from a diagonal vector.

Zero diagonal entries are treated as fixed-0 dofs: their inverse is set to zero.
"""
function jacobi_precond(diag::AbstractVector{T}) where {T<:Number}
    invd = similar(diag)
    z = zero(T)
    @inbounds for i in eachindex(diag)
        d = diag[i]
        invd[i] = d == z ? z : inv(d)
    end
    return Diagonal(invd)
end

raw"""Compute a discrete $L^2$ error by comparing samples on a point set.

This utility is intended for examples and sanity checks.

There are three variants:

1. **High-level**: `discrete_l2_error(u, grid, basis, uex, ref_level; eval_nodes=...)`.
   Builds a tensor-product reference point set from `eval_nodes` and `ref_level`,
   evaluates the sparse grid expansion, and calls the medium-level variant.
2. **Medium-level**: `discrete_l2_error(u, grid, basis, uex, eval_points)`.
   Evaluates the sparse grid expansion at `eval_points` via the evaluation interface
   (`plan_evaluate` + `evaluate`) and calls the low-level variant.
3. **Low-level**: `discrete_l2_error(uvals, uex, eval_points)`.
   Here `uvals` is interpreted as the *already evaluated* vector aligned with
   `eval_points`.

All variants return `(L2, relL2)` where

```math
L2 = \sqrt{\frac{\sum_i |u_h(x_i) - u_{ex}(x_i)|^2}{n_{\mathrm{pts}}}},\qquad
\mathrm{relL2} = \sqrt{\frac{\sum_i |u_h(x_i) - u_{ex}(x_i)|^2}{\sum_i |u_{ex}(x_i)|^2}}.
```
"""
function discrete_l2_error(uvals::AbstractVector,
                           uex,
                           eval_points::AbstractPointSet{D};
                           T::Union{Nothing,Type{<:Real}}=nothing) where {D}
    length(uvals) == length(eval_points) ||
        throw(DimensionMismatch("uvals has length $(length(uvals)) but expected $(length(eval_points))"))

    Tr = T === nothing ? _realpart_type(eltype(uvals)) : T
    err2 = zero(Tr)
    ue2 = zero(Tr)

    if eval_points isa TensorProductPoints{D}
        P = eval_points
        lengths = ntuple(d -> length(P.pts[d]), Val(D))
        I = CartesianIndices(lengths)
        lin = LinearIndices(I)
        Tp = eltype(P.pts[1])
        @inbounds for J in I
            xpt = SVector{D,Tp}(ntuple(d -> P.pts[d][J[d]], Val(D)))
            ue = uex(xpt)
            e = uvals[lin[J]] - ue
            err2 += convert(Tr, abs2(e))
            ue2 += convert(Tr, abs2(ue))
        end
    elseif eval_points isa ScatteredPoints{D}
        P = eval_points
        Tp = eltype(P.X)
        @inbounds for i in 1:length(P)
            xpt = SVector{D,Tp}(ntuple(d -> P.X[d, i], Val(D)))
            ue = uex(xpt)
            e = uvals[i] - ue
            err2 += convert(Tr, abs2(e))
            ue2 += convert(Tr, abs2(ue))
        end
    else
        throw(ArgumentError("Unsupported point set type: $(typeof(eval_points))"))
    end

    L2 = sqrt(err2 / Tr(length(uvals)))
    relL2 = sqrt(err2 / ue2)
    return L2, relL2
end

function discrete_l2_error(u::AbstractVector,
                           grid::SparseGrid{<:SparseGridSpec{D}},
                           bases::Union{AbstractUnivariateBasis,NTuple{D,AbstractUnivariateBasis}},
                           uex,
                           eval_points::AbstractPointSet{D};
                           T::Union{Nothing,Type{<:Real}}=nothing,
                           backend::Union{Symbol,AbstractEvaluationBackend}=:auto,
                           Ti::Type{<:Integer}=Int) where {D}
    length(u) == length(grid) || throw(DimensionMismatch("u must have length(grid)"))
    plan = plan_evaluate(grid, bases, eval_points, eltype(u); backend=backend, Ti=Ti)
    uvals = evaluate(plan, u)
    return discrete_l2_error(uvals, uex, eval_points; T=T)
end

function discrete_l2_error(u::AbstractVector,
                           grid::SparseGrid{<:SparseGridSpec{D}},
                           basis::AbstractUnivariateBasis,
                           uex,
                           ref_level::Integer;
                           T::Union{Nothing,Type{<:Real}}=nothing,
                           eval_nodes::AbstractUnivariateNodes=ChebyshevGaussLobattoNodes(NaturalOrder(); endpoints=false)) where {D}
    ref_level < 0 && throw(ArgumentError("ref_level must satisfy ref_level ≥ 0, got ref_level=$ref_level"))
    length(u) == length(grid) || throw(DimensionMismatch("u must have length(grid)"))

    xref = points(eval_nodes, Int(ref_level))
    Pref = TensorProductPoints(ntuple(_ -> xref, Val(D)))
    return discrete_l2_error(u, grid, basis, uex, Pref; T=T)
end

# -----------------------------------------------------------------------------
# Hat-basis Helmholtz line operators (Balder–Zenger)

raw"""1D hierarchical-hat mass operator.

This is the 1D mass operator in the **hierarchical hat basis** used by the
Balder–Zenger sparse Galerkin discretization.

It is implemented in a matrix-free, in-place form via the decomposition

```math
M = D\,(Y^{-1} - E) + (Y^{-T} - E)\,D + \tfrac23 D,
```

where

- $Y$ is the (dyadic) hierarchization transform,
- $D$ is a simple diagonal scaling in hierarchical order,
- $E$ keeps only endpoint contributions.

The `Part` type parameter is a symbol:

- `:full`  : the full operator $M$ (allocates a small scratch buffer).
- `:lower` : the lower-triangular contribution $D(Y^{-1} - E) + \tfrac23 D$.
- `:upper` : the upper-triangular contribution $(Y^{-T} - E)D$.

This supports `UpDownTensorOp` via `updown(op)`.
"""
struct HatMassLineOp{Part,T} <: AbstractLineOp
    hdiag::Vector{Vector{T}}   # h_i per coefficient, per level (hierarchical ordering)
end

needs_plan(::Type{<:HatMassLineOp}) = Val(true)

function lineplan(::HatMassLineOp, nodes::DyadicNodes{<:LevelOrder,EndpointMask{0x3}}, rmax::Integer, ::Type{T}) where {T}
    L = Int(rmax)
    plan = DyadicHierarchyShared(nodes, L, Int)
    return fill(plan, L + 1)
end

function lineplan(::HatMassLineOp, ::DyadicNodes{<:LevelOrder,EndpointMask{0x0}}, ::Integer, ::Type)
    throw(ArgumentError("HatMassLineOp requires dyadic nodes with endpoints"))
end

function HatMassLineOp(nodes::DyadicNodes{<:Any,EndpointMask{0x3}}, rmax::Int;
                       part::Symbol=:full,
                       T::Type{<:Real}=Float64)
    hdiag = Vector{Vector{T}}(undef, rmax + 1)
    @inbounds for lvl in 0:rmax
        n = totalsize(nodes, lvl)
        h = Vector{T}(undef, n)

        h[1] = T(1) / T(2)
        h[2] = T(1) / T(2)

        for ℓ in 1:lvl
            hℓ = one(T) / T(1 << ℓ)
            m = 1 << (ℓ - 1)
            start = m + 2
            for r in 0:(m - 1)
                h[start + r] = hℓ
            end
        end

        hdiag[lvl + 1] = h
    end

    if part === :full
        return HatMassLineOp{:full,T}(hdiag)
    elseif part === :lower
        return HatMassLineOp{:lower,T}(hdiag)
    elseif part === :upper
        return HatMassLineOp{:upper,T}(hdiag)
    else
        throw(ArgumentError("invalid part=$part (expected :full, :lower, or :upper)"))
    end
end

HatMassLineOp(nodes::DyadicNodes{<:Any,EndpointMask{0x0}}, ::Int; kwargs...) =
    throw(ArgumentError("HatMassLineOp requires dyadic nodes with endpoints"))

function updown(op::HatMassLineOp{:full,T}) where {T}
    L = HatMassLineOp{:lower,T}(op.hdiag)
    U = HatMassLineOp{:upper,T}(op.hdiag)
    return L, U
end
updown(op::HatMassLineOp{:lower}) = (op, ZeroLineOp())
updown(op::HatMassLineOp{:upper}) = (ZeroLineOp(), op)

"""1D stiffness diagonal in the hierarchical hat basis (Balder–Zenger Eq. (9)-(10))."""
struct HatStiffnessDiagLineOp{T} <: AbstractLineOp
    diags::Vector{Vector{T}}
end

function HatStiffnessDiagLineOp(nodes::DyadicNodes{<:Any,EndpointMask{0x3}}, rmax::Int;
                                T::Type{<:Real}=Float64)
    diags = Vector{Vector{T}}(undef, rmax + 1)
    @inbounds for lvl in 0:rmax
        n = totalsize(nodes, lvl)
        d = Vector{T}(undef, n)

        d[1] = one(T)
        d[2] = one(T)

        for ℓ in 1:lvl
            sℓ = T(2) * T(1 << ℓ)
            m = 1 << (ℓ - 1)
            start = m + 2
            for r in 0:(m - 1)
                d[start + r] = sℓ
            end
        end

        diags[lvl + 1] = d
    end

    return HatStiffnessDiagLineOp(diags)
end

HatStiffnessDiagLineOp(nodes::DyadicNodes{<:Any,EndpointMask{0x0}}, ::Int; kwargs...) =
    throw(ArgumentError("HatStiffnessDiagLineOp requires dyadic nodes with endpoints"))

updown(op::HatStiffnessDiagLineOp) = (op, ZeroLineOp())

@inline function apply_line!(out::AbstractVector{T}, op::HatStiffnessDiagLineOp{T}, inp::AbstractVector{T}, work::AbstractVector{T},
                             ::DyadicNodes, level::Int, plan) where {T}
    d = op.diags[level + 1]
    length(d) == length(inp) || throw(DimensionMismatch("diag length mismatch at refinement index=$level"))
    @inbounds for i in eachindex(inp)
        out[i] = d[i] * inp[i]
    end
    return out
end

function apply_line!(out::AbstractVector{T}, op::HatMassLineOp{:lower,T}, inp::AbstractVector{T}, work::AbstractVector{T},
                     nodes::DyadicNodes, level::Int, plan::DyadicHierarchyShared) where {T}
    h = op.hdiag[level + 1]
    length(h) == length(inp) || throw(DimensionMismatch("hdiag length mismatch at refinement index=$level"))

    copyto!(out, inp)
    _dyadic_dehierarchize!(out, nodes, _dyadic_aux(plan, level))

    c23 = T(2) / T(3)
    @inbounds for i in eachindex(inp)
        out[i] = h[i] * (out[i] - inp[i]) + c23 * h[i] * inp[i]
    end

    length(inp) >= 2 && (@inbounds out[2] += (T(1) / T(6)) * inp[1])
    return out
end

function apply_line!(out::AbstractVector{T}, op::HatMassLineOp{:upper,T}, inp::AbstractVector{T}, work::AbstractVector{T},
                     nodes::DyadicNodes, level::Int, plan::DyadicHierarchyShared) where {T}
    h = op.hdiag[level + 1]
    length(h) == length(inp) || throw(DimensionMismatch("hdiag length mismatch at refinement index=$level"))

    @inbounds for i in eachindex(inp)
        out[i] = h[i] * inp[i]
    end
    _dyadic_dehierarchize_transpose!(out, nodes, _dyadic_aux(plan, level))

    @inbounds for i in eachindex(inp)
        out[i] -= h[i] * inp[i]
    end

    length(inp) >= 2 && (@inbounds out[1] += (T(1) / T(6)) * inp[2])
    return out
end

function apply_line!(out::AbstractVector{T}, op::HatMassLineOp{:full,T}, inp::AbstractVector{T}, work::AbstractVector{T},
                     nodes::DyadicNodes, level::Int, plan::DyadicHierarchyShared) where {T}
    h = op.hdiag[level + 1]
    length(h) == length(inp) || throw(DimensionMismatch("hdiag length mismatch at refinement index=$level"))

    tmp = @view work[1:length(inp)]

    @inbounds for i in eachindex(inp)
        tmp[i] = h[i] * inp[i]
    end
    _dyadic_dehierarchize_transpose!(tmp, nodes, _dyadic_aux(plan, level))
    @inbounds for i in eachindex(inp)
        tmp[i] -= h[i] * inp[i]
    end

    copyto!(out, inp)
    _dyadic_dehierarchize!(out, nodes, _dyadic_aux(plan, level))
    c23 = T(2) / T(3)
    @inbounds for i in eachindex(inp)
        out[i] = h[i] * (out[i] - inp[i]) + c23 * h[i] * inp[i]
    end

    @inbounds for i in eachindex(out)
        out[i] += tmp[i]
    end

    if length(inp) >= 2
        @inbounds out[1] += (T(1) / T(6)) * inp[2]
        @inbounds out[2] += (T(1) / T(6)) * inp[1]
    end

    return out
end
