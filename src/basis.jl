"""Univariate modal bases.

This file provides a small abstraction layer for 1D bases that appear as the
modal (coefficient) spaces in sparse grid algorithms.

The basis interface is intentionally minimal:

* [`ncoeff`](@ref): number of coefficients at a given node rule and level.
* [`vandermonde`](@ref): evaluation matrix at arbitrary points.
"""

# -----------------------------------------------------------------------------
# Basis interface

"""Abstract supertype for all 1D bases."""
abstract type AbstractUnivariateBasis end

"""Number of coefficients for `basis` at axis family `axis` and refinement index `r`.

Default: matches the axis size, i.e. `totalsize(axis, r)`.
"""
ncoeff(::AbstractUnivariateBasis, axis::AbstractAxisFamily, r::Integer) = totalsize(axis, r)

"""Construct a basis evaluation (Vandermonde) matrix.

`V` has size `(length(x), ncoeff)` with entries

    V[i,k] = ϕ_{k-1}(x[i]),    k = 1..ncoeff.
"""
vandermonde(::AbstractUnivariateBasis, ::AbstractVector, ::Integer; T::Type{<:Number}=Float64) =
    throw(ArgumentError("vandermonde not implemented for this basis"))

# -----------------------------------------------------------------------------
# Concrete bases

struct ChebyshevTBasis <: AbstractUnivariateBasis end
struct LegendreBasis <: AbstractUnivariateBasis end
struct DirichletLegendreBasis <: AbstractUnivariateBasis end
struct FourierBasis <: AbstractUnivariateBasis end

# -----------------------------------------------------------------------------
# Vandermonde builders (explicit recurrences; correctness-first baseline)

function vandermonde(::ChebyshevTBasis, x::AbstractVector{<:Real}, n::Integer;
                     T::Type{<:Number}=Float64)
    m = length(x)
    n < 0 && throw(ArgumentError("ncoeff must be ≥ 0"))
    V = Matrix{T}(undef, m, n)
    n == 0 && return V
    @inbounds for i in 1:m
        xi = T(x[i])
        V[i, 1] = one(T)
        if n >= 2
            V[i, 2] = xi
            tkm1 = one(T)
            tk = xi
            for k in 3:n
                tkp1 = 2 * xi * tk - tkm1
                V[i, k] = tkp1
                tkm1 = tk
                tk = tkp1
            end
        end
    end
    return V
end

function vandermonde(::LegendreBasis, x::AbstractVector{<:Real}, n::Integer;
                     T::Type{<:Number}=Float64)
    m = length(x)
    n < 0 && throw(ArgumentError("ncoeff must be ≥ 0"))
    V = Matrix{T}(undef, m, n)
    n == 0 && return V
    @inbounds for i in 1:m
        xi = T(x[i])
        V[i, 1] = one(T)
        if n >= 2
            V[i, 2] = xi
            pkm1 = one(T)
            pk = xi
            for k in 3:n
                nn = k - 2
                pkp1 = ((2nn + 1) * xi * pk - nn * pkm1) / (nn + 1)
                V[i, k] = pkp1
                pkm1 = pk
                pk = pkp1
            end
        end
    end
    return V
end

function vandermonde(::DirichletLegendreBasis, x::AbstractVector{<:Real}, n::Integer;
                     T::Type{<:Number}=Float64)
    n < 0 && throw(ArgumentError("ncoeff must be ≥ 0"))
    # Need P_k up to k = (n-1) + 2.
    Vleg = vandermonde(LegendreBasis(), x, n + 2; T=T)
    m = size(Vleg, 1)
    V = Matrix{T}(undef, m, n)
    @inbounds for i in 1:m
        for k in 1:n
            V[i, k] = Vleg[i, k] - Vleg[i, k + 2]
        end
    end
    return V
end

function vandermonde(::FourierBasis, t::AbstractVector{<:Real}, n::Integer;
                     T::Type{<:Complex}=ComplexF64)
    m = length(t)
    n < 0 && throw(ArgumentError("ncoeff must be ≥ 0"))
    V = Matrix{T}(undef, m, n)
    n == 0 && return V
    twoπim = T(2) * T(im) * T(pi)
    @inbounds for i in 1:m
        ti = t[i]
        for k in 1:n
            V[i, k] = exp(twoπim * (k - 1) * ti)
        end
    end
    return V
end

# -----------------------------------------------------------------------------
# Local-support bases (hierarchical, compact support)

"""Piecewise-linear hierarchical hat basis on dyadic grids.

The basis adapts to the node rule: if the dyadic nodes include endpoints at
level 0, the basis includes the corresponding boundary functions; otherwise,
level 0 is empty.
"""
struct HatBasis <: AbstractUnivariateBasis end

"""Piecewise-polynomial hierarchical basis on dyadic grids.

At level `ℓ`, the (local) polynomial degree defaults to `p = min(ℓ+1, pmax)`.
"""
struct PiecewisePolynomialBasis <: AbstractUnivariateBasis
    pmax::Int
end
PiecewisePolynomialBasis(; pmax::Int=typemax(Int)) = PiecewisePolynomialBasis(pmax)

# Measures / supports
measure(::HatBasis) = UnitIntervalMeasure()
measure(::PiecewisePolynomialBasis) = UnitIntervalMeasure()

# -----------------------------------------------------------------------------
# Support-style trait + local micro-interface

abstract type AbstractSupportStyle end
struct GlobalSupport <: AbstractSupportStyle end
struct LocalSupport <: AbstractSupportStyle end

"""Return a trait describing whether the basis functions have global or local support."""
support_style(::AbstractUnivariateBasis) = GlobalSupport()
support_style(::HatBasis) = LocalSupport()
support_style(::PiecewisePolynomialBasis) = LocalSupport()

"""Return the (0-based) local indices in level `ℓ` with nonzero basis value at `x`.

For most local bases (hat / piecewise polynomial) this is either empty or a
singleton tuple.
"""
active_local(::AbstractUnivariateBasis, ::AbstractUnivariateNodes, ::Integer, ::Real) =
    throw(ArgumentError("active_local not implemented for this basis/nodes"))

"""Evaluate a local basis function (identified by `ℓ` and 0-based `local`) at `x`."""
eval_local(::AbstractUnivariateBasis, ::AbstractUnivariateNodes, ::Integer, ::Integer, ::Real) =
    throw(ArgumentError("eval_local not implemented for this basis/nodes"))

# -- Dyadic hat basis ----------------------------------------------------------

@inline function _dyadic_hat_center_index(nodes::DyadicNodes{NaturalOrder}, ℓ::Integer, loc::Integer)
    # Odd index i = 2*loc + 1
    return Int(2 * loc + 1)
end

@inline function _dyadic_hat_center_index(nodes::DyadicNodes{LevelOrder}, ℓ::Integer, loc::Integer)
    # Odd index i = 2*loc + 1 (same local ordering as NaturalOrder).
    return Int(2 * loc + 1)
end

@inline function _dyadic_hat_local_from_cell(nodes::DyadicNodes{NaturalOrder}, ℓ::Integer, a::Integer)
    return Int(a)
end

@inline function _dyadic_hat_local_from_cell(nodes::DyadicNodes{LevelOrder}, ℓ::Integer, a::Integer)
    return Int(a)
end

function active_local(::HatBasis, nodes::DyadicNodes{<:Any,EM}, ℓ::Integer, x::Real) where {EM<:AbstractEndpointMode}
    (x < 0 || x > 1) && return ()
    if ℓ == 0
        if has_left(EM) && has_right(EM)
            return (0, 1)
        elseif n_endpoints(EM) == 1
            return (0,)
        else
            return ()
        end
    end
    ℓ < 0 && return ()
    m = 1 << ℓ
    # Cell index j in 0..m-1
    j = floor(Int, m * x)
    j = clamp(j, 0, m - 1)
    a = j >> 1               # a in 0..2^(ℓ-1)-1
    loc = _dyadic_hat_local_from_cell(nodes, ℓ, a)
    return (loc,)
end

function eval_local(::HatBasis, nodes::DyadicNodes{<:Any,EM}, ℓ::Integer, loc::Integer, x::Real) where {EM<:AbstractEndpointMode}
    Tf = promote_type(typeof(x), Float64)
    xf = Tf(float(x))

    if ℓ == 0
        if has_left(EM) && has_right(EM)
            loc == 0 && return one(Tf) - xf
            loc == 1 && return xf
            throw(BoundsError("HatBasis level-0 local index must be 0 or 1"))
        elseif has_left(EM)
            loc == 0 && return one(Tf) - xf
            throw(BoundsError("HatBasis level-0 local index must be 0"))
        elseif has_right(EM)
            loc == 0 && return xf
            throw(BoundsError("HatBasis level-0 local index must be 0"))
        else
            throw(ArgumentError("HatBasis has no level-0 functions for nodes without endpoints"))
        end
    end

    ℓ < 0 && return zero(Tf)
    i = _dyadic_hat_center_index(nodes, ℓ, loc)
    # ϕ_{ℓ,i}(x) = max(1 - |2^ℓ x - i|, 0)
    t = abs((2.0^ℓ) * xf - i)
    return t < 1 ? (1 - t) : zero(Tf)
end

# -- Piecewise polynomial basis -----------------------------------------------

@inline function _piecewise_poly_degree(b::PiecewisePolynomialBasis, ℓ::Integer)
    return max(0, min(b.pmax, Int(ℓ) + 1))
end

"""Select additional zero locations (beyond the two support endpoints) for the
piecewise polynomial basis.

This is a simple, deterministic stencil: we add zeros at odd offsets ±3, ±5, …
times `h_ℓ` around the center, skipping candidates outside `[0,1]`, and using
one-sided candidates near the boundary.
"""
function _piecewise_poly_zeros(xc::Float64, h::Float64, p::Int)
    # Always include the two support endpoints.
    zeros = Float64[xc - h, xc + h]
    need = p - 2
    need <= 0 && return zeros

    # Add offsets in increasing distance from the center.
    k = 1
    while length(zeros) < p
        off = (2k + 1) * h
        zl = xc - off
        zr = xc + off
        if 0.0 <= zl <= 1.0
            push!(zeros, zl)
            length(zeros) == p && break
        end
        if 0.0 <= zr <= 1.0
            push!(zeros, zr)
            length(zeros) == p && break
        end
        k += 1
        # Safety: should never happen for p = ℓ+1 with dyadic h, but avoid infinite loops.
        k > 10_000 && break
    end

    # If we still don't have enough (very close to a boundary), fall back to
    # allowing boundary points.
    while length(zeros) < p
        # Prefer the nearest boundary.
        push!(zeros, (xc < 0.5) ? 1.0 : 0.0)
    end

    return zeros
end

function active_local(::PiecewisePolynomialBasis, nodes::DyadicNodes, ℓ::Integer, x::Real)
    # Same active-index logic as hats: disjoint supports at fixed level.
    return active_local(HatBasis(), nodes, ℓ, x)
end

function eval_local(b::PiecewisePolynomialBasis, nodes::DyadicNodes, ℓ::Integer, loc::Integer, x::Real)
    ℓ == 0 && return eval_local(HatBasis(), nodes, 0, loc, x)
    ℓ < 0 && return 0.0
    xf = float(x)
    (xf < 0.0 || xf > 1.0) && return 0.0

    i = _dyadic_hat_center_index(nodes, ℓ, loc)
    h = 2.0^(-Int(ℓ))
    xc = i * h
    # Compact support.
    (xf < xc - h || xf > xc + h) && return 0.0

    p = _piecewise_poly_degree(b, ℓ)
    p <= 1 && return eval_local(HatBasis(), nodes, ℓ, loc, xf)

    zeros = _piecewise_poly_zeros(xc, h, p)

    # Lagrange-form polynomial with specified roots.
    num = 1.0
    den = 1.0
    @inbounds for z in zeros
        num *= (xf - z)
        den *= (xc - z)
    end
    return num / den
end

# -----------------------------------------------------------------------------
# Dense Vandermonde (correctness / testing fallback)

function vandermonde(::HatBasis, x::AbstractVector{<:Real}, n::Integer;
                     T::Type{<:Number}=Float64)
    n < 0 && throw(ArgumentError("ncoeff must be ≥ 0"))
    m = length(x)
    V = Matrix{T}(undef, m, n)
    n == 0 && return V

    if ispow2(Int(n) + 1)
        # No endpoints: levels 1..L with total count n = 2^L - 1.
        L = round(Int, log2(n + 1))
        (1 << L) == (n + 1) || throw(ArgumentError("HatBasis expects ncoeff = 2^L - 1 (no endpoints)"))

        # Column enumeration: level 1 has 1 fn, level 2 has 2, level 3 has 4, ...
        col = 1
        @inbounds for ℓ in 1:L
            nℓ = 1 << (ℓ - 1)
            for a in 0:(nℓ - 1)
                i = 2a + 1
                for r in 1:m
                    t = abs((2.0^ℓ) * float(x[r]) - i)
                    V[r, col] = t < 1 ? (1 - t) : zero(T)
                end
                col += 1
            end
        end
        return V
    elseif n >= 2 && ispow2(Int(n) - 1)
        # Both endpoints: 2 boundary fns + HatBasis up to level L.
        L = round(Int, log2(n - 1))
        (1 << L) == (n - 1) || throw(ArgumentError("HatBasis expects ncoeff = 2^L + 1 (with endpoints)"))

        @inbounds for r in 1:m
            xr = float(x[r])
            V[r, 1] = 1 - xr
            V[r, 2] = xr
        end

        col = 3
        @inbounds for ℓ in 1:L
            nℓ = 1 << (ℓ - 1)
            for a in 0:(nℓ - 1)
                i = 2a + 1
                for r in 1:m
                    t = abs((2.0^ℓ) * float(x[r]) - i)
                    V[r, col] = t < 1 ? (1 - t) : zero(T)
                end
                col += 1
            end
        end
        return V
    end

    throw(ArgumentError("HatBasis expects ncoeff = 2^L - 1 (no endpoints) or 2^L + 1 (with endpoints), got ncoeff=$n"))
end

function vandermonde(b::PiecewisePolynomialBasis, x::AbstractVector{<:Real}, n::Integer;
                     T::Type{<:Number}=Float64)
    n < 0 && throw(ArgumentError("ncoeff must be ≥ 0"))
    m = length(x)
    V = Matrix{T}(undef, m, n)
    n == 0 && return V

    col0 = if ispow2(Int(n) + 1)
        # No endpoints: levels 1..L with total count n = 2^L - 1.
        L = round(Int, log2(n + 1))
        (1 << L) == (n + 1) || throw(ArgumentError("PiecewisePolynomialBasis expects ncoeff = 2^L - 1 (no endpoints)"))
        1
    elseif n >= 2 && ispow2(Int(n) - 1)
        # Both endpoints: 2 boundary fns + PiecewisePolynomialBasis up to level L.
        L = round(Int, log2(n - 1))
        (1 << L) == (n - 1) || throw(ArgumentError("PiecewisePolynomialBasis expects ncoeff = 2^L + 1 (with endpoints)"))

        @inbounds for r in 1:m
            xr = float(x[r])
            V[r, 1] = 1 - xr
            V[r, 2] = xr
        end
        3
    else
        throw(ArgumentError("PiecewisePolynomialBasis expects ncoeff = 2^L - 1 (no endpoints) or 2^L + 1 (with endpoints), got ncoeff=$n"))
    end

    # Same level structure as HatBasis (interior levels only).
    if col0 == 1
        L = round(Int, log2(n + 1))
    else
        L = round(Int, log2(n - 1))
    end

    col = col0
    @inbounds for ℓ in 1:L
        nℓ = 1 << (ℓ - 1)
        h = 2.0^(-ℓ)
        p = _piecewise_poly_degree(b, ℓ)
        for a in 0:(nℓ - 1)
            i = 2a + 1
            xc = i * h
            zeros = (p <= 1) ? Float64[] : _piecewise_poly_zeros(xc, h, p)
            for r in 1:m
                xr = float(x[r])
                if xr < xc - h || xr > xc + h
                    V[r, col] = zero(T)
                elseif p <= 1
                    t = abs((2.0^ℓ) * xr - i)
                    V[r, col] = t < 1 ? (1 - t) : zero(T)
                else
                    num = 1.0
                    den = 1.0
                    for z in zeros
                        num *= (xr - z)
                        den *= (xc - z)
                    end
                    V[r, col] = num / den
                end
            end
            col += 1
        end
    end
    return V
end

