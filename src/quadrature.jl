# Adaptive sparse quadrature

# -----------------------------------------------------------------------------
# Rule / family layer

abstract type AbstractQuadratureFamily end
abstract type AbstractNestedQuadratureFamily <: AbstractQuadratureFamily end
abstract type AbstractQuadraturePointFamily end
abstract type AbstractNestedQuadraturePointFamily <: AbstractQuadraturePointFamily end

"""A concrete 1D quadrature rule."""
struct QuadratureRule{XT<:Real,WT<:Number,DT<:Integer}
    points::Vector{XT}
    weights::Vector{WT}
    degree::DT
    function QuadratureRule(points::AbstractVector{XT}, weights::AbstractVector{WT}, degree::DT) where {XT<:Real,WT<:Number,DT<:Integer}
        length(points) == length(weights) || throw(DimensionMismatch("points/weights length mismatch"))
        return new{XT,WT,DT}(collect(points), collect(weights), degree)
    end
end

Base.length(rule::QuadratureRule) = length(rule.points)
qpoints(rule::QuadratureRule) = rule.points
qweights(rule::QuadratureRule) = rule.weights
qdegree(rule::QuadratureRule) = rule.degree

"""Optional measure/weight associated with a quadrature family."""
qmeasure(::AbstractQuadratureFamily) = nothing
"""Degree/exactness metadata of a 1D quadrature family at refinement index `r`."""
qdegree(Q::AbstractQuadratureFamily, r::Integer) = qdegree(qrule(Q, r))
"""Full 1D quadrature rule at refinement index `r`."""
qrule(Q::AbstractQuadratureFamily, r::Integer) = error("qrule not implemented for $(typeof(Q))")
"""Incremental 1D difference rule Δ_r.

For non-nested families the default is the full rule `qrule(Q, r)`;
nested families should specialize this to return the true incremental difference rule.
"""
qdiffrule(Q::AbstractQuadratureFamily, r::Integer) = qrule(Q, r)

"""Number of points in the 1D quadrature rule at refinement index `r`."""
qsize(Q::AbstractQuadratureFamily, r::Integer) = length(qrule(Q, r))
qpoints(Q::AbstractQuadratureFamily, r::Integer) = qpoints(qrule(Q, r))
qweights(Q::AbstractQuadratureFamily, r::Integer) = qweights(qrule(Q, r))

is_nested(::AbstractQuadratureFamily) = false
is_nested(::AbstractNestedQuadratureFamily) = true

# current base families
struct ClenshawCurtisQuadrature{T<:Real} <: AbstractNestedQuadratureFamily end
ClenshawCurtisQuadrature(::Type{T}=Float64) where {T<:Real} = ClenshawCurtisQuadrature{T}()
qmeasure(::ClenshawCurtisQuadrature) = LegendreMeasure()

struct GaussLegendreQuadrature{T<:Real} <: AbstractQuadratureFamily end
GaussLegendreQuadrature(::Type{T}=Float64) where {T<:Real} = GaussLegendreQuadrature{T}()
qmeasure(::GaussLegendreQuadrature) = LegendreMeasure()

struct GaussLaguerreQuadrature{T<:Real,S<:Real} <: AbstractQuadratureFamily
    α::S
end
GaussLaguerreQuadrature(α::Real=0.0, ::Type{T}=Float64) where {T<:Real} = GaussLaguerreQuadrature{T,typeof(float(α))}(float(α))
qmeasure(Q::GaussLaguerreQuadrature{T,S}) where {T,S} = LaguerreMeasure(Q.α)

struct GaussHermiteQuadrature{T<:Real} <: AbstractQuadratureFamily end
GaussHermiteQuadrature(::Type{T}=Float64) where {T<:Real} = GaussHermiteQuadrature{T}()
qmeasure(::GaussHermiteQuadrature) = HermiteMeasure()

QuadratureFamily(Q::AbstractQuadratureFamily) = Q
QuadratureFamily(::ChebyshevGaussLobattoNodes) = ClenshawCurtisQuadrature()
QuadratureFamily(::GaussNodes{<:LegendreMeasure}) = GaussLegendreQuadrature()
QuadratureFamily(n::GaussNodes{<:LaguerreMeasure}) = GaussLaguerreQuadrature(measure(n).α)
QuadratureFamily(::GaussNodes{<:HermiteMeasure}) = GaussHermiteQuadrature()

# -----------------------------------------------------------------------------
# Basis/measure-aware point/rule families

# make-shift point families (nested, 1 point per refinement index)
mutable struct WeightedLejaPoints{B,M,T<:Real} <: AbstractNestedQuadraturePointFamily
    basis::B
    measure::M
    domain::Tuple{T,T}
    candidate_count::Int
    cache::Dict{Int,Vector{T}}
end

function WeightedLejaPoints(basis::AbstractUnivariateBasis=LegendreBasis(),
                            measure::AbstractMeasure=LegendreMeasure();
                            T::Type{<:Real}=Float64,
                            domain::Tuple{<:Real,<:Real}= _finite_domain(measure, T),
                            candidate_count::Integer=129)
    return WeightedLejaPoints(basis, measure, (T(domain[1]), T(domain[2])), Int(candidate_count), Dict{Int,Vector{T}}())
end

mutable struct WeightedLejaQuadrature{B,M,T<:Real} <: AbstractNestedQuadratureFamily
    points::WeightedLejaPoints{B,M,T}
    rule_cache::Dict{Int,QuadratureRule{T,T,Int}}
    diff_cache::Dict{Int,QuadratureRule{T,T,Int}}
end

function WeightedLejaQuadrature(basis::AbstractUnivariateBasis=LegendreBasis(),
                                measure::AbstractMeasure=LegendreMeasure();
                                T::Type{<:Real}=Float64,
                                kwargs...)
    pts = WeightedLejaPoints(basis, measure; T=T, kwargs...)
    return WeightedLejaQuadrature(
        pts,
        Dict{Int,QuadratureRule{T,T,Int}}(),
        Dict{Int,QuadratureRule{T,T,Int}}(),
    )
end
qmeasure(Q::WeightedLejaQuadrature) = Q.points.measure

mutable struct PseudoGaussQuadrature{B,M,T<:Real} <: AbstractNestedQuadratureFamily
    basis::B
    measure::M
    domain::Tuple{T,T}
    candidate_count::Int
    point_cache::Dict{Int,Vector{T}}
    rule_cache::Dict{Int,QuadratureRule{T,T,Int}}
    diff_cache::Dict{Int,QuadratureRule{T,T,Int}}
end

function PseudoGaussQuadrature(basis::AbstractUnivariateBasis=LegendreBasis(),
                               measure::AbstractMeasure=LegendreMeasure();
                               T::Type{<:Real}=Float64,
                               domain::Tuple{<:Real,<:Real}= _finite_domain(measure, T),
                               candidate_count::Integer=129)
    return PseudoGaussQuadrature(
        basis, measure, (T(domain[1]), T(domain[2])), Int(candidate_count),
        Dict{Int,Vector{T}}(), Dict{Int,QuadratureRule{T,T,Int}}(), Dict{Int,QuadratureRule{T,T,Int}}())
end
qmeasure(Q::PseudoGaussQuadrature) = Q.measure

struct IdentityMap end
(::IdentityMap)(x) = x
struct ReciprocalExpMap{T<:Real}
    α::T
end
# maps [0,∞) -> (0,1)
(::ReciprocalExpMap{T})(x) where {T} = one(T) - exp(-x)
reciprocalexp_jac(x, α::T) where {T} = exp(-x)

struct MappedGaussianQuadrature{Q<:AbstractQuadratureFamily,F,J,M} <: AbstractQuadratureFamily
    base::Q
    map::F
    jacobian::J
    measure::M
end
MappedGaussianQuadrature(base::Q; map=IdentityMap(), jacobian=(x->one(x)), measure=qmeasure(base)) where {Q<:AbstractQuadratureFamily} =
    MappedGaussianQuadrature{Q,typeof(map),typeof(jacobian),typeof(measure)}(base, map, jacobian, measure)
qmeasure(Q::MappedGaussianQuadrature) = Q.measure
is_nested(Q::MappedGaussianQuadrature) = is_nested(Q.base)
qsize(Q::MappedGaussianQuadrature, r::Integer) = qsize(Q.base, r)

# -----------------------------------------------------------------------------
# helper utilities

@inline _check_refinement(r::Integer) = (r < 0) && throw(ArgumentError("refinement index must satisfy r ≥ 0, got $r"))

function _finite_domain(::LegendreMeasure, ::Type{T}) where {T<:Real}
    return (T(-1), T(1))
end
function _finite_domain(::ChebyshevTMeasure, ::Type{T}) where {T<:Real}
    return (T(-1), T(1))
end
function _finite_domain(::ChebyshevUMeasure, ::Type{T}) where {T<:Real}
    return (T(-1), T(1))
end
function _finite_domain(::JacobiMeasure, ::Type{T}) where {T<:Real}
    return (T(-1), T(1))
end
function _finite_domain(::UnitIntervalMeasure, ::Type{T}) where {T<:Real}
    return (zero(T), one(T))
end
function _finite_domain(::FourierMeasure, ::Type{T}) where {T<:Real}
    return (zero(T), one(T))
end
function _finite_domain(m::LaguerreMeasure, ::Type{T}) where {T<:Real}
    n = 64
    x, _ = _gauss_laguerre_nodes_weights(n, T, T(m.α))
    return (zero(T), maximum(x))
end
function _finite_domain(::HermiteMeasure, ::Type{T}) where {T<:Real}
    n = 64
    x, _ = _gauss_hermite_nodes_weights(n, T)
    return (minimum(x), maximum(x))
end

_measure_weight(::LegendreMeasure, x) = one(x)
_measure_weight(::UnitIntervalMeasure, x) = one(x)
_measure_weight(::FourierMeasure, x) = one(x)
_measure_weight(::ChebyshevTMeasure, x) = inv(sqrt(max(eps(real(x)), one(x) - x*x)))
_measure_weight(::ChebyshevUMeasure, x) = sqrt(max(zero(x), one(x) - x*x))
_measure_weight(m::JacobiMeasure, x) = (one(x)-x)^m.α * (one(x)+x)^m.β
_measure_weight(m::LaguerreMeasure, x) = x^m.α * exp(-x)
_measure_weight(::HermiteMeasure, x) = exp(-x*x)

# -----------------------------------------------------------------------------
# classical 1D rules

qdegree(::ClenshawCurtisQuadrature, r::Integer) = qsize(ClenshawCurtisQuadrature(), r) - 1
qsize(::ClenshawCurtisQuadrature, r::Integer) = (_check_refinement(r); (1 << Int(r)) + 1)

function qrule(::ClenshawCurtisQuadrature{T}, r::Integer) where {T<:Real}
    n = qsize(ClenshawCurtisQuadrature(T), r)
    return QuadratureRule(qpoints(ClenshawCurtisQuadrature(T), r), qweights(ClenshawCurtisQuadrature(T), r), n - 1)
end

function qpoints(::ClenshawCurtisQuadrature{T}, r::Integer) where {T<:Real}
    n = qsize(ClenshawCurtisQuadrature(T), r)
    N = n - 1
    return T.(cos.(T(pi) .* collect(0:N) ./ T(N)))
end

function qweights(::ClenshawCurtisQuadrature{T}, r::Integer) where {T<:Real}
    n = qsize(ClenshawCurtisQuadrature(T), r)
    N = n - 1
    N == 1 && return T[one(T), one(T)]
    θ = T(pi) .* collect(0:N) ./ T(N)
    w = zeros(T, n)
    v = ones(T, N - 1)
    ii = 2:N
    if iseven(N)
        w1 = inv(T(N)^2 - one(T))
        w[1] = w1; w[end] = w1
        for k in 1:(N ÷ 2 - 1)
            c = (T(2) / T(4k^2 - 1)) .* cos.(T(2k) .* θ[ii])
            @. v = v - c
        end
        @. v = v - cos(T(N) * θ[ii]) / (T(N)^2 - one(T))
    else
        w1 = inv(T(N)^2)
        w[1] = w1; w[end] = w1
        for k in 1:((N - 1) ÷ 2)
            c = (T(2) / T(4k^2 - 1)) .* cos.(T(2k) .* θ[ii])
            @. v = v - c
        end
    end
    @. w[ii] = T(2) * v / T(N)
    return w
end

function _gauss_legendre_nodes_weights(n::Integer, ::Type{T}) where {T<:Real}
    n > 0 || throw(ArgumentError("n must be positive, got n=$n"))
    n == 1 && return (T[zero(T)], T[T(2)])
    β = Vector{T}(undef, n - 1)
    @inbounds for k in 1:(n - 1)
        β[k] = T(k) / sqrt(T(4k^2 - 1))
    end
    J = SymTridiagonal(zeros(T, n), β)
    E = eigen(J)
    x = E.values
    V = E.vectors
    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        w[i] = T(2) * abs2(V[1, i])
    end
    return x, w
end

function _gauss_laguerre_nodes_weights(n::Integer, ::Type{T}, α::T) where {T<:Real}
    n > 0 || throw(ArgumentError("n must be positive, got n=$n"))
    β = Vector{T}(undef, n)
    γ = Vector{T}(undef, n-1)
    @inbounds for k in 1:n
        β[k] = T(2k - 1) + α
    end
    @inbounds for k in 1:n-1
        γ[k] = sqrt(T(k) * (T(k) + α))
    end
    J = SymTridiagonal(β, γ)
    E = eigen(J)
    x = E.values
    V = E.vectors
    w = Vector{T}(undef, n)
    Γ = gamma(α + one(T))
    @inbounds for i in 1:n
        w[i] = Γ * abs2(V[1, i])
    end
    return x, w
end

function _gauss_hermite_nodes_weights(n::Integer, ::Type{T}) where {T<:Real}
    n > 0 || throw(ArgumentError("n must be positive, got n=$n"))
    if n == 1
        return (T[zero(T)], T[sqrt(T(pi))])
    end
    β = Vector{T}(undef, n-1)
    @inbounds for k in 1:n-1
        β[k] = sqrt(T(k) / T(2))
    end
    J = SymTridiagonal(zeros(T, n), β)
    E = eigen(J)
    x = E.values
    V = E.vectors
    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        w[i] = sqrt(T(pi)) * abs2(V[1, i])
    end
    return x, w
end

qdegree(::GaussLegendreQuadrature, r::Integer) = 2*qsize(GaussLegendreQuadrature(), r) - 1
qsize(::GaussLegendreQuadrature, r::Integer) = (_check_refinement(r); (1 << Int(r)) + 1)
function qrule(::GaussLegendreQuadrature{T}, r::Integer) where {T<:Real}
    n = qsize(GaussLegendreQuadrature(T), r)
    x, w = _gauss_legendre_nodes_weights(n, T)
    return QuadratureRule(x, w, 2n - 1)
end

qdegree(Q::GaussLaguerreQuadrature{T,S}, r::Integer) where {T<:Real,S<:Real} = 2*qsize(Q, r) - 1
qsize(::GaussLaguerreQuadrature, r::Integer) = (_check_refinement(r); (1 << Int(r)) + 1)
function qrule(Q::GaussLaguerreQuadrature{T,S}, r::Integer) where {T<:Real,S<:Real}
    n = qsize(Q, r)
    x, w = _gauss_laguerre_nodes_weights(n, T, T(Q.α))
    return QuadratureRule(x, w, 2n - 1)
end

qdegree(::GaussHermiteQuadrature, r::Integer) = 2*qsize(GaussHermiteQuadrature(), r) - 1
qsize(::GaussHermiteQuadrature, r::Integer) = (_check_refinement(r); (1 << Int(r)) + 1)
function qrule(::GaussHermiteQuadrature{T}, r::Integer) where {T<:Real}
    n = qsize(GaussHermiteQuadrature(T), r)
    x, w = _gauss_hermite_nodes_weights(n, T)
    return QuadratureRule(x, w, 2n - 1)
end

# direct difference rules
function qdiffrule(Q::ClenshawCurtisQuadrature{T}, r::Integer) where {T<:Real}
    _check_refinement(r)
    hi = qrule(Q, r)
    r == 0 && return hi
    lo = qrule(Q, r - 1)
    w = copy(qweights(hi))
    @views w[1:2:end] .-= qweights(lo)
    nz = map(!iszero, w)
    return QuadratureRule(qpoints(hi)[nz], w[nz], qdegree(hi))
end

function qdiffrule(Q::AbstractNestedQuadratureFamily, r::Integer)
    _check_refinement(r)
    hi = qrule(Q, r)
    r == 0 && return hi
    lo = qrule(Q, r - 1)
    pts_hi = qpoints(hi)
    w_hi = qweights(hi)
    w = copy(w_hi)
    # old points are assumed nested exactly; subtract old weights where they appear
    for (xlo, wlo) in zip(qpoints(lo), qweights(lo))
        j = findfirst(x -> x == xlo, pts_hi)
        j === nothing && throw(ArgumentError("nested quadrature family $(typeof(Q)) produced inconsistent point sets"))
        w[j] -= wlo
    end
    nz = map(!iszero, w)
    return QuadratureRule(pts_hi[nz], w[nz], qdegree(hi))
end


# -----------------------------------------------------------------------------
# Weighted Leja points and quadrature rules

qsize(::WeightedLejaPoints, r::Integer) = (_check_refinement(r); Int(r) + 1)

function _candidate_grid(domain::Tuple{T,T}, m::Integer) where {T<:Real}
    a, b = domain
    m <= 1 && return T[(a+b)/2]
    ks = collect(0:m-1)
    return T.((a+b)/2 .+ (b-a)/2 .* cos.(pi .* ks ./ (m-1)))
end

function _basis_vector(b::AbstractUnivariateBasis, x::AbstractVector{T}, n::Integer) where {T}
    return vandermonde(b, x, n; T=T)
end

function _weighted_leja_points!(W::WeightedLejaPoints{B,M,T}, n::Integer) where {B,M,T}
    haskey(W.cache, n) && return W.cache[n]
    pts = T[]
    if n == 1
        a,b = W.domain
        x0 = if isfinite(a) && isfinite(b)
            (a+b)/2
        elseif W.measure isa LaguerreMeasure
            zero(T)
        else
            zero(T)
        end
        pts = T[x0]
        W.cache[1] = pts
        return pts
    end
    if haskey(W.cache, n-1)
        pts = copy(W.cache[n-1])
    else
        pts = copy(_weighted_leja_points!(W, n-1))
    end
    k = n
    cand = _candidate_grid(W.domain, max(W.candidate_count, 16n + 1))
    # exclude current points approximately
    bestx = cand[1]
    bestv = -one(T)
    for x in cand
        any(y -> abs(y - x) <= sqrt(eps(T)), pts) && continue
        if k == 1
            score = sqrt(abs(_measure_weight(W.measure, x)))
        else
            Vprev = _basis_vector(W.basis, pts, k - 1)
            rhs = _basis_vector(W.basis, T[x], k)[1, k]
            vxc = vec(_basis_vector(W.basis, T[x], k - 1))
            vals = _basis_vector(W.basis, pts, k)[:, k]
            c = Vprev \ vals
            residual = rhs - dot(vxc, c)
            score = abs(residual) * sqrt(abs(_measure_weight(W.measure, x)))
        end
        if score > bestv
            bestv = score
            bestx = x
        end
    end
    push!(pts, bestx)
    W.cache[n] = pts
    return pts
end

qpoints(W::WeightedLejaPoints, r::Integer) = _weighted_leja_points!(W, qsize(W, r))

function _rule_moments_from_measure(basis::AbstractUnivariateBasis, measure::AbstractMeasure, n::Integer, ::Type{T}) where {T<:Real}
    # Compute ∫ φ_j dμ for j=0:n-1 using a sufficiently accurate reference quadrature.
    refN = max(2n + 4, 16)
    rr = if measure isa LegendreMeasure
        QuadratureRule(_gauss_legendre_nodes_weights(refN, T)..., 2refN - 1)
    elseif measure isa LaguerreMeasure
        QuadratureRule(_gauss_laguerre_nodes_weights(refN, T, T(measure.α))..., 2refN - 1)
    elseif measure isa HermiteMeasure
        QuadratureRule(_gauss_hermite_nodes_weights(refN, T)..., 2refN - 1)
    elseif measure isa UnitIntervalMeasure
        # mapped Gauss-Legendre
        x, w = _gauss_legendre_nodes_weights(refN, T)
        QuadratureRule((x .+ one(T)) ./ T(2), w ./ T(2), 2refN - 1)
    else
        # fallback: Clenshaw–Curtis on finite domain with weight
        dom = _finite_domain(measure, T)
        pts = _candidate_grid(dom, refN)
        # composite trapezoid fallback
        Δ = (dom[2] - dom[1]) / (refN - 1)
        ww = fill(T(Δ), refN)
        ww[1] /= 2; ww[end] /= 2
        QuadratureRule(pts, ww .* _measure_weight.(Ref(measure), pts), refN - 1)
    end
    V = vandermonde(basis, qpoints(rr), n; T=T)
    m = Vector{T}(undef, n)
    @inbounds for j in 1:n
        m[j] = dot(qweights(rr), view(V, :, j))
    end
    return m
end

function _quadrature_weights_from_points(points::AbstractVector{T}, basis::AbstractUnivariateBasis, measure::AbstractMeasure) where {T<:Real}
    n = length(points)
    V = vandermonde(basis, points, n; T=T)
    m = _rule_moments_from_measure(basis, measure, n, T)
    # solve V' w = m
    w = Matrix(V') \ m
    return Vector{T}(w)
end

function qrule(Q::WeightedLejaQuadrature{B,M,T}, r::Integer) where {B,M,T}
    _check_refinement(r)
    n = Int(r) + 1
    if haskey(Q.rule_cache, n)
        return Q.rule_cache[n]
    end
    pts = qpoints(Q.points, r)
    w = _quadrature_weights_from_points(pts, Q.points.basis, Q.points.measure)
    rule = QuadratureRule(pts, w, n - 1)
    Q.rule_cache[n] = rule
    return rule
end
qdegree(Q::WeightedLejaQuadrature, r::Integer) = qsize(Q, r) - 1
qsize(::WeightedLejaQuadrature, r::Integer) = (_check_refinement(r); Int(r) + 1)

function qdiffrule(Q::WeightedLejaQuadrature{B,M,T}, r::Integer) where {B,M,T}
    _check_refinement(r)
    n = Int(r) + 1
    if haskey(Q.diff_cache, n)
        return Q.diff_cache[n]
    end
    hi = qrule(Q, r)
    if r == 0
        Q.diff_cache[n] = hi
        return hi
    end
    lo = qrule(Q, r - 1)
    w = copy(qweights(hi))
    @views w[1:length(lo)] .-= qweights(lo)
    nz = map(!iszero, w)
    rule = QuadratureRule(qpoints(hi)[nz], w[nz], qdegree(hi))
    Q.diff_cache[n] = rule
    return rule
end

# -----------------------------------------------------------------------------
# Pseudo-Gauss quadrature

function _gram_targets(basis::AbstractUnivariateBasis, measure::AbstractMeasure, n::Integer, ::Type{T}) where {T<:Real}
    refN = max(2n + 8, 32)
    rr = if measure isa LegendreMeasure
        QuadratureRule(_gauss_legendre_nodes_weights(refN, T)..., 2refN - 1)
    elseif measure isa LaguerreMeasure
        QuadratureRule(_gauss_laguerre_nodes_weights(refN, T, T(measure.α))..., 2refN - 1)
    elseif measure isa HermiteMeasure
        QuadratureRule(_gauss_hermite_nodes_weights(refN, T)..., 2refN - 1)
    elseif measure isa UnitIntervalMeasure
        x, w = _gauss_legendre_nodes_weights(refN, T)
        QuadratureRule((x .+ one(T)) ./ T(2), w ./ T(2), 2refN - 1)
    else
        dom = _finite_domain(measure, T)
        pts = _candidate_grid(dom, refN)
        Δ = (dom[2] - dom[1]) / (refN - 1)
        ww = fill(T(Δ), refN)
        ww[1] /= 2; ww[end] /= 2
        QuadratureRule(pts, ww .* _measure_weight.(Ref(measure), pts), refN - 1)
    end
    mmax = 2n - 1
    V = vandermonde(basis, qpoints(rr), mmax + 1; T=T)
    pairs = Tuple{Int,Int}[]
    g = T[]
    for m in 0:n-1
        for k in m:(2n - 1 - m)
            push!(pairs, (m+1, k+1))
            push!(g, dot(qweights(rr), V[:,m+1] .* V[:,k+1]))
        end
    end
    return pairs, g
end

function _pseudogauss_weights_residual(points::AbstractVector{T}, basis::AbstractUnivariateBasis, measure::AbstractMeasure) where {T<:Real}
    n = length(points)
    pairs, g = _gram_targets(basis, measure, n, T)
    V = vandermonde(basis, points, 2n; T=T)
    P = Matrix{T}(undef, length(pairs), n)
    @inbounds for (row, (i,j)) in enumerate(pairs)
        for c in 1:n
            P[row,c] = V[c,i] * V[c,j]
        end
    end
    w = P \ g
    res = norm(P * w - g)
    return Vector{T}(w), res
end

function _pseudogauss_points!(Q::PseudoGaussQuadrature{B,M,T}, n::Integer) where {B,M,T}
    haskey(Q.point_cache, n) && return Q.point_cache[n]
    if n == 1
        x0 = if Q.measure isa LegendreMeasure || Q.measure isa HermiteMeasure
            zero(T)
        else
            (Q.domain[1] + Q.domain[2]) / 2
        end
        pts = T[x0]
        Q.point_cache[1] = pts
        return pts
    end
    pts_old = haskey(Q.point_cache, n-1) ? copy(Q.point_cache[n-1]) : copy(_pseudogauss_points!(Q, n-1))
    cand = _candidate_grid(Q.domain, max(Q.candidate_count, 24n + 1))
    bestx = cand[1]
    bestres = T(Inf)
    for x in cand
        any(y -> abs(y - x) <= sqrt(eps(T)), pts_old) && continue
        pts = [pts_old; x]
        _, res = _pseudogauss_weights_residual(pts, Q.basis, Q.measure)
        if res < bestres
            bestres = res
            bestx = x
        end
    end
    pts = [pts_old; bestx]
    Q.point_cache[n] = pts
    return pts
end

function qrule(Q::PseudoGaussQuadrature{B,M,T}, r::Integer) where {B,M,T}
    _check_refinement(r)
    n = Int(r) + 1
    if haskey(Q.rule_cache, n)
        return Q.rule_cache[n]
    end
    pts = _pseudogauss_points!(Q, n)
    w, _ = _pseudogauss_weights_residual(pts, Q.basis, Q.measure)
    rule = QuadratureRule(pts, w, 2n - 1)
    Q.rule_cache[n] = rule
    return rule
end
qdegree(Q::PseudoGaussQuadrature, r::Integer) = 2*qsize(Q, r) - 1
qsize(::PseudoGaussQuadrature, r::Integer) = (_check_refinement(r); Int(r) + 1)

function qdiffrule(Q::PseudoGaussQuadrature{B,M,T}, r::Integer) where {B,M,T}
    _check_refinement(r)
    n = Int(r) + 1
    if haskey(Q.diff_cache, n)
        return Q.diff_cache[n]
    end
    hi = qrule(Q, r)
    if r == 0
        Q.diff_cache[n] = hi
        return hi
    end
    lo = qrule(Q, r - 1)
    w = copy(qweights(hi))
    @views w[1:length(lo)] .-= qweights(lo)
    nz = map(!iszero, w)
    rule = QuadratureRule(qpoints(hi)[nz], w[nz], qdegree(hi))
    Q.diff_cache[n] = rule
    return rule
end

# -----------------------------------------------------------------------------
# mapped families

function qrule(Q::MappedGaussianQuadrature, r::Integer)
    base = qrule(Q.base, r)
    pts = map(Q.map, qpoints(base))
    w = similar(qweights(base), promote_type(eltype(qweights(base)), eltype(pts)))
    @inbounds for i in eachindex(w)
        x = qpoints(base)[i]
        w[i] = qweights(base)[i] * Q.jacobian(x)
    end
    return QuadratureRule(pts, w, qdegree(base))
end
qdiffrule(Q::MappedGaussianQuadrature, r::Integer) = begin
    base = qdiffrule(Q.base, r)
    pts = map(Q.map, qpoints(base))
    w = similar(qweights(base), promote_type(eltype(qweights(base)), eltype(pts)))
    @inbounds for i in eachindex(w)
        # jacobian on mapped points uses same preimage because qdiffrule points are in base coordinates
        x = qpoints(base)[i]
        w[i] = qweights(base)[i] * Q.jacobian(x)
    end
    QuadratureRule(pts, w, qdegree(base))
end
qdegree(Q::MappedGaussianQuadrature, r::Integer) = qdegree(Q.base, r)

# -----------------------------------------------------------------------------
# Tensor-difference bookkeeping

struct FrontierEntry{R,CT<:Number,ET<:Real,PT<:Real}
    r::R
    contribution::CT
    estimate::ET
    priority::PT
end

function Base.isless(a::FrontierEntry{R,CT,ET,PT}, b::FrontierEntry{R,CT,ET,PT}) where {R,CT<:Number,ET<:Real,PT<:Real}
    isless(a.priority, b.priority) && return true
    isless(b.priority, a.priority) && return false
    return isless(Tuple(a.r), Tuple(b.r))
end

mutable struct AdaptiveQuadratureState{R,CT<:Number,ET<:Real,PT<:Real,HeapT,StatusT,PendingT}
    heap::HeapT
    status::StatusT
    pending::PendingT

    integral::CT
    eta::ET

    nevals::Int
    ncalls::Int
    work::Float64

    nactive::Int
    nold::Int

    accepted::Vector{R}
    frontier_pops::Vector{R}
    accepted_contrib::Vector{CT}
    refinement_hist::Vector{Vector{Int}}
end

Base.length(state::AdaptiveQuadratureState) = state.nactive + state.nold
Base.isempty(state::AdaptiveQuadratureState) = isempty(state.heap)

@inline _estimate(value, nevals::Integer, work::Real, mode::Symbol) = abs(value)

@inline function _priority(value, nevals::Integer, work::Real, mode::Symbol)
    aval = abs(value)
    if mode === :absdelta
        return aval
    elseif mode === :profit
        return aval / max(work, 1.0)
    elseif mode === :normalized
        return aval / max(nevals, 1)
    else
        throw(ArgumentError("unsupported indicator=$mode (use :absdelta/:profit/:normalized)"))
    end
end

# -----------------------------------------------------------------------------
# tensor-difference contribution kernels

struct TensorRuleKernel{D,XT,WT,AT,PT<:NTuple{D,AbstractVector},WTT<:NTuple{D,AbstractVector}}
    pts::PT
    wts::WTT
end

function TensorRuleKernel(pts::PT,
                          wts::WTT,
                          sample::AT) where {D,PT<:NTuple{D,AbstractVector},WTT<:NTuple{D,AbstractVector},AT}
    XT = promote_type(map(eltype, pts)...)
    WT = promote_type(map(eltype, wts)...)
    return TensorRuleKernel{D,XT,WT,AT,PT,WTT}(pts, wts)
end

@generated function _tensor_rule_integral_chunk(integrand,
                                                kernel::TensorRuleKernel{D,XT,WT,AT,PT,WTT},
                                                outer_lo::Int,
                                                outer_hi::Int,
                                                ::Val{SKIPFIRST}) where {D,XT,WT,AT,PT,WTT,SKIPFIRST}
    pts = :pts
    wts = :wts

    if D == 1
        inner_lo = SKIPFIRST ? :(nextind($pts[1], outer_lo)) : :outer_lo
        return quote
            pts = kernel.pts
            wts = kernel.wts
            acc = zero(AT)
            @inbounds begin
                inner_lo = $inner_lo
                @simd for i_1 = inner_lo:outer_hi
                    x_1 = XT(pts[1][i_1])
                    acc += convert(AT, integrand(SVector{1,XT}(x_1)) * wts[1][i_1])
                end
            end
            return acc
        end
    end

    outer_ranges = Expr(:tuple,
                        [d == 1 ? :(outer_lo:outer_hi) : :(firstindex($pts[$d]):lastindex($pts[$d])) for d in 1:(D - 1)]...)
    x_assigns = [:( $(Symbol("x_$(d)")) = XT($pts[$d][$(Symbol("i_$(d)"))]) ) for d in 1:(D - 1)]
    w_assigns = Any[:(w_1 = $wts[1][i_1])]
    append!(w_assigns, [:( $(Symbol("w_$(d)")) = $(Symbol("w_$(d - 1)")) * $wts[$d][$(Symbol("i_$(d)"))] ) for d in 2:(D - 1)])
    skip_conds = [:( $(Symbol("i_$(d)")) == firstindex($pts[$d]) ) for d in 1:(D - 1)]
    skip_cond = reduce((a, b) -> :($a && $b), skip_conds)
    inner_lo = SKIPFIRST ? :($skip_cond ? nextind($pts[$D], firstindex($pts[$D])) : firstindex($pts[$D])) : :(firstindex($pts[$D]))
    inner_i = Symbol("i_$(D)")
    inner_x = Symbol("x_$(D)")
    point = Expr(:call,
                 Expr(:curly, :SVector, D, :XT),
                 [Symbol("x_$(d)") for d in 1:D]...)
    weight = :($(Symbol("w_$(D - 1)")) * $wts[$D][$inner_i])

    return quote
        pts = kernel.pts
        wts = kernel.wts
        ranges = $outer_ranges
        acc = zero(AT)
        @inbounds begin
            Base.Cartesian.@nloops $(D - 1) i d -> ranges[d] begin
                $(Expr(:block, x_assigns...))
                $(Expr(:block, w_assigns...))
                inner_lo = $inner_lo
                @simd for $inner_i = inner_lo:lastindex(pts[$D])
                    $inner_x = XT(pts[$D][$inner_i])
                    acc += convert(AT, integrand($point) * $weight)
                end
            end
        end
        return acc
    end
end

function _tensor_rule_integral(integrand,
                               rules::NTuple{D,QuadratureRule};
                               threaded::Bool=true) where {D}
    pts = ntuple(d -> qpoints(rules[d]), Val(D))
    wts = ntuple(d -> qweights(rules[d]), Val(D))
    XT = promote_type(map(eltype, pts)...)
    WT = promote_type(map(eltype, wts)...)

    npts = 1
    @inbounds for d in 1:D
        npts *= length(pts[d])
    end

    first_outer = firstindex(pts[1])
    last_outer = lastindex(pts[1])
    firsts = ntuple(d -> firstindex(pts[d]), Val(D))
    x0 = SVector{D,XT}(ntuple(d -> XT(pts[d][firsts[d]]), Val(D)))
    w0 = one(WT)
    @inbounds for d in 1:D
        w0 *= wts[d][firsts[d]]
    end
    seed = integrand(x0) * w0
    kernel = TensorRuleKernel(pts, wts, seed)
    acc = seed
    acc += _tensor_rule_integral_chunk(integrand, kernel, first_outer, first_outer, Val(true))

    remaining_lo = nextind(pts[1], first_outer)
    if remaining_lo <= last_outer
        remaining_outer = last_outer - remaining_lo + 1
        use_threads = threaded && nthreads() > 1 && npts >= 4096 && remaining_outer >= 2
        if use_threads
            nchunks = min(nthreads(), remaining_outer)
            chunklen = cld(remaining_outer, nchunks)
            tasks = Task[]
            for k in 1:nchunks
                lo = remaining_lo + (k - 1) * chunklen
                hi = min(last_outer, lo + chunklen - 1)
                lo <= hi || continue
                push!(tasks, @spawn _tensor_rule_integral_chunk(integrand, kernel, lo, hi, Val(false)))
            end
            for task in tasks
                acc += fetch(task)
            end
        else
            acc += _tensor_rule_integral_chunk(integrand, kernel, remaining_lo, last_outer, Val(false))
        end
    end

    return acc, npts
end

"""Tensor-difference quadrature contribution and work stats for `Δ_r`.

Returns `(acc, nevals, work)`.
"""
function delta_contribution(integrand,
                            families::NTuple{D,AbstractQuadratureFamily},
                            r::SVector{D,<:Integer}) where {D}
    rS = SVector{D,Int}(r)
    any(<(0), rS) && throw(ArgumentError("refinement index must be nonnegative, got $rS"))

    if all(is_nested, families)
        rules = ntuple(d -> qdiffrule(families[d], rS[d]), Val(D))
        acc, nevals = _tensor_rule_integral(integrand, rules)
        return acc, nevals, float(nevals)
    end

    valid = Tuple{Int,SVector{D,Int}}[]
    @inbounds for mask in 0:((1 << D) - 1)
        rr = ntuple(d -> rS[d] - ((mask >> (d - 1)) & 0x1), Val(D))
        any(<(0), rr) && continue
        push!(valid, (mask, SVector{D,Int}(rr)))
    end
    isempty(valid) && throw(ArgumentError("no tensor-difference contributions found for refinement index $rS"))

    function reduce_masks(lo::Int, hi::Int)
        mask0, rr0 = valid[lo]
        rules0 = ntuple(d -> qrule(families[d], rr0[d]), Val(D))
        acc_local, nevals_local = _tensor_rule_integral(integrand, rules0; threaded=false)
        if isodd(count_ones(mask0))
            acc_local = -acc_local
        end
        work_local = float(nevals_local)
        @inbounds for idx in (lo + 1):hi
            mask, rrS = valid[idx]
            rules = ntuple(d -> qrule(families[d], rrS[d]), Val(D))
            val, npts = _tensor_rule_integral(integrand, rules; threaded=false)
            nevals_local += npts
            work_local += npts
            if isodd(count_ones(mask))
                acc_local -= val
            else
                acc_local += val
            end
        end
        return acc_local, nevals_local, work_local
    end

    if nthreads() > 1 && length(valid) > 1
        nchunks = min(nthreads(), length(valid))
        chunklen = cld(length(valid), nchunks)
        tasks = Task[]
        for k in 1:nchunks
            lo = (k - 1) * chunklen + 1
            hi = min(length(valid), lo + chunklen - 1)
            lo <= hi || continue
            push!(tasks, @spawn reduce_masks(lo, hi))
        end
        acc, nevals, work = fetch(first(tasks))
        for task in Iterators.drop(tasks, 1)
            acc_local, nevals_local, work_local = fetch(task)
            acc += acc_local
            nevals += nevals_local
            work += work_local
        end
        return acc, nevals, work
    end

    return reduce_masks(1, length(valid))
end

# -----------------------------------------------------------------------------
# adaptive integration

function _normalize_qfamilies(family::AbstractQuadratureFamily, ::Val{D}) where {D}
    return ntuple(_ -> family, Val(D))
end
function _normalize_qfamilies(families::NTuple{D,<:AbstractQuadratureFamily}, ::Val{D}) where {D}
    return families
end
function _normalize_qfamilies(families::Tuple, ::Val{D}) where {D}
    length(families) == D || throw(DimensionMismatch("quadrature family tuple length mismatch"))
    return ntuple(d -> begin
        Q = families[d]
        Q isa AbstractQuadratureFamily || throw(ArgumentError("families[$d] is not an AbstractQuadratureFamily"))
        Q
    end, Val(D))
end

"""Dimension-adaptive sparse quadrature over a static admissibility envelope.

Returns `(integral, state)` where `state` follows the Gerstner–Griebel
active/old semantics:

- `integral` already includes contributions from both active and old indices,
- `eta` is the additive estimate over the active set only,
- `accepted` records activation order, and
- `frontier_pops` records the order in which active indices are moved to old.
"""
function integrate_adaptive(integrand,
                            families::NTuple{D,AbstractQuadratureFamily},
                            envelope::AbstractIndexSet{D};
                            atol::Real=0,
                            rtol::Real=sqrt(eps(Float64)),
                            maxterms::Int=10_000,
                            maxevals::Int=typemax(Int),
                            indicator::Symbol=:absdelta) where {D}
    maxterms > 0 || throw(ArgumentError("maxterms must be positive, got $maxterms"))
    maxevals > 0 || throw(ArgumentError("maxevals must be positive, got $maxevals"))

    r0 = SVector{D,Int}(ntuple(_ -> 0, Val(D)))
    contains(envelope, r0) || throw(ArgumentError("envelope must contain the zero refinement index"))

    c0, nevals0, work0 = delta_contribution(integrand, families, r0)
    CT = typeof(c0)
    ET = typeof(_estimate(c0, nevals0, work0, indicator))
    PT = typeof(_priority(c0, nevals0, work0, indicator))
    R = typeof(r0)
    EntryT = FrontierEntry{R,CT,ET,PT}

    c = convert(CT, c0)
    est = convert(ET, _estimate(c, nevals0, work0, indicator))
    pri = convert(PT, _priority(c, nevals0, work0, indicator))

    heap = BinaryMaxHeap{EntryT}()
    push!(heap, EntryT(r0, c, est, pri))
    status = RobinDict{R,Bool}()
    status[r0] = true
    pending = RobinDict{R,UInt16}()
    state = AdaptiveQuadratureState{R,CT,ET,PT,typeof(heap),typeof(status),typeof(pending)}(
        heap,
        status,
        pending,
        c,
        est,
        nevals0,
        1,
        work0,
        1,
        0,
        R[r0],
        R[],
        CT[c],
        [Int[r0[d]] for d in 1:D],
    )

    while !isempty(state.heap) && length(state) < maxterms && state.nevals < maxevals
        tol = ET(atol) + ET(rtol) * max(one(ET), ET(abs(state.integral)))
        state.eta <= tol && break

        popped = false
        while !isempty(state.heap)
            entry = pop!(state.heap)
            get(state.status, entry.r, false) || continue

            state.status[entry.r] = false
            state.eta -= entry.estimate
            state.nactive -= 1
            state.nold += 1
            push!(state.frontier_pops, entry.r)
            popped = true

            @inbounds for j in 1:D
                length(state) >= maxterms && break
                state.nevals >= maxevals && break

                cand = setindex(entry.r, entry.r[j] + 1, j)
                contains(envelope, cand) || continue
                haskey(state.status, cand) && continue

                count_old = get(state.pending, cand, UInt16(0x00)) + UInt16(0x01)
                needed = UInt16(count(!iszero, cand))
                count_old > needed && throw(ArgumentError("candidate predecessor count exceeded admissibility count for $cand"))
                if count_old != needed
                    state.pending[cand] = count_old
                    continue
                end
                state.pending[cand] = count_old

                acc, nevals, work = delta_contribution(integrand, families, cand)
                c = convert(CT, acc)
                est = convert(ET, _estimate(c, nevals, work, indicator))
                pri = convert(PT, _priority(c, nevals, work, indicator))
                push!(state.heap, EntryT(cand, c, est, pri))
                state.status[cand] = true
                delete!(state.pending, cand)
                state.integral += c
                state.eta += est
                state.nevals += nevals
                state.ncalls += 1
                state.work += work
                state.nactive += 1
                push!(state.accepted, cand)
                push!(state.accepted_contrib, c)
                @inbounds for d in 1:D
                    push!(state.refinement_hist[d], cand[d])
                end
            end
            break
        end
        popped || throw(ArgumentError("frontier is empty"))
    end

    return state.integral, state
end

function integrate_adaptive(integrand,
                            families,
                            envelope::AbstractIndexSet{D}; kwargs...) where {D}
    Q = _normalize_qfamilies(families, Val(D))
    return integrate_adaptive(integrand, Q, envelope; kwargs...)
end
