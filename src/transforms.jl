"""1D transforms and hierarchization kernels.

This file collects the 1D building blocks used by the unidirectional sparse-grid
engine:

- plan-based nodal ↔ modal line transforms (FFT, DCT-I via FFTW r2r)
- Chebyshev ↔ Legendre coefficient conversion (FastTransforms connection plans)
- Dirichlet–Legendre basis kernels (O(n))
- size-based hierarchization / dehierarchization sweeps and their plan wrappers
"""

# -----------------------------------------------------------------------------
# Utilities and shared planning hooks

@inline _realpart_type(::Type{T}) where {T<:Number} = typeof(real(zero(T)))

make_plan_shared(op, axis, rmax, ::Type{T}) where {T} = nothing
make_plan_entry(op, axis, n::Integer, r::Integer, ::Type{T}, shared) where {T} =
    throw(ArgumentError("make_plan_entry not implemented for $(typeof(op)) on $(typeof(axis))"))


function _apply_plan!(::Val{:mul}, plan, buf::AbstractVector, work::AbstractVector)
    lmul!(plan, buf)
    return buf
end

function _apply_plan!(::Val{:div}, plan, buf::AbstractVector, work::AbstractVector)
    ldiv!(plan, buf)
    return buf
end

_apply_plan!(action, plan, buf::AbstractVector, work::AbstractVector) =
    throw(ArgumentError("_apply_plan! not implemented for action=$action and plan $(typeof(plan))"))

# -----------------------------------------------------------------------------
# Size / refinement helpers

function _cgl_levelorder_indices_from_size(n::Integer)
    n == 2 && return [0, 1]
    n == 3 && return [0, 2, 1]
    N = Int(n) - 1
    child = _cgl_levelorder_indices_from_size((Int(n) + 1) >>> 1)
    odd = collect(1:2:(N - 1))
    return vcat(2 .* child, odd)
end

# -----------------------------------------------------------------------------
# Nodal ↔ modal line transforms (in-place, plan-based)

"""Read-only Fourier nodal ↔ modal transform plan of size `n`.

Storage contract:
- input nodal values are in the nodal storage order (typically `LevelOrder()`),
- output modal coefficients are in `NaturalOrder()`.

`lmul!` applies nodal → modal; `ldiv!` applies modal → nodal.
"""
struct FourierPlan{Tr,Pf,Pb}
    perm::Vector{Int}
    fwd::Pf
    bwd::Pb
end

function FourierPlan(axis::FourierEquispacedNodes, n::Integer, ::Type{T}) where {T<:Number}
    N = Int(n)
    Tr = _realpart_type(T)
    perm = nodeorder(axis) isa LevelOrder ? bitrevperm(N) : collect(0:(N - 1))
    probe = Vector{Complex{Tr}}(undef, N)
    fwd = FastTransforms.FFTW.plan_fft!(probe; flags=FastTransforms.FFTW.ESTIMATE)
    bwd = FastTransforms.FFTW.plan_ifft!(probe; flags=FastTransforms.FFTW.ESTIMATE)
    return FourierPlan{Tr,typeof(fwd),typeof(bwd)}(perm, fwd, bwd)
end

"""Read-only Chebyshev–Gauss–Lobatto nodal ↔ modal transform plan of size `n`.

Storage contract:
- input nodal values are in the nodal storage order (typically `LevelOrder()`),
- output modal coefficients are in `NaturalOrder()`.

`lmul!` applies nodal → modal; `ldiv!` applies modal → nodal.
"""
struct ChebyshevPlan{Tr,Pf,Pb}
    idx::Vector{Int}
    fwd::Pf
    bwd::Pb
end

function ChebyshevPlan(axis::ChebyshevGaussLobattoNodes{<:AbstractNodeOrder,EndpointMask{0x3}}, n::Integer, ::Type{T}) where {T<:Number}
    idx = nodeorder(axis) isa LevelOrder ? _cgl_levelorder_indices_from_size(Int(n)) : collect(0:(Int(n) - 1))
    Tr = _realpart_type(T)
    probe = Vector{Tr}(undef, Int(n))
    fwd = FastTransforms.FFTW.plan_r2r!(probe, FastTransforms.FFTW.REDFT00; flags=FastTransforms.FFTW.ESTIMATE)
    bwd = fwd
    return ChebyshevPlan{Tr,typeof(fwd),typeof(bwd)}(idx, fwd, bwd)
end

function ChebyshevPlan(::ChebyshevGaussLobattoNodes{<:AbstractNodeOrder,EM}, ::Integer, ::Type) where {EM<:AbstractEndpointMode}
    throw(ArgumentError("ChebyshevPlan (DCT-I) is only implemented for nodes with both endpoints"))
end

"""Plan for converting Chebyshev (T) coefficients ↔ Legendre (P) coefficients.

The plan acts on coefficient vectors of length `n` corresponding to degrees
`0:(n-1)`.

- `lmul!(P, x)` applies Chebyshev → Legendre.
- `ldiv!(P, x)` applies Legendre → Chebyshev.
"""
struct ChebyshevLegendrePlan{Tr,Pf,Pb}
    fwd::Pf
    bwd::Pb
end

function ChebyshevLegendrePlan(n::Integer, ::Type{T}=Float64) where {T<:Number}
    nn = Int(n)
    Tr = _realpart_type(T)
    fwd = FastTransforms.plan_cheb2leg(Tr, nn; normcheb=false, normleg=false)
    bwd = FastTransforms.plan_leg2cheb(Tr, nn; normleg=false, normcheb=false)
    return ChebyshevLegendrePlan{Tr,typeof(fwd),typeof(bwd)}(fwd, bwd)
end

function LinearAlgebra.lmul!(P::FourierPlan{Tr}, x::AbstractVector{Complex{Tr}}) where {Tr}
    work = similar(x)
    _apply_plan!(Val(:mul), P, x, work)
end

function LinearAlgebra.ldiv!(P::FourierPlan{Tr}, x::AbstractVector{Complex{Tr}}) where {Tr}
    work = similar(x)
    _apply_plan!(Val(:div), P, x, work)
end

Base.:*(P::FourierPlan, x::AbstractVector) = (y = copy(x); lmul!(P, y); y)
Base.:\(P::FourierPlan, x::AbstractVector) = (y = copy(x); ldiv!(P, y); y)

function LinearAlgebra.lmul!(P::ChebyshevPlan{T}, x::AbstractVector{T}) where {T<:Real}
    work = similar(x)
    _apply_plan!(Val(:mul), P, x, work)
end

function LinearAlgebra.ldiv!(P::ChebyshevPlan{T}, x::AbstractVector{T}) where {T<:Real}
    work = similar(x)
    _apply_plan!(Val(:div), P, x, work)
end

function LinearAlgebra.lmul!(P::ChebyshevPlan{Tr}, x::AbstractVector{Complex{Tr}}) where {Tr<:Real}
    work = similar(x)
    _apply_plan!(Val(:mul), P, x, work)
end

function LinearAlgebra.ldiv!(P::ChebyshevPlan{Tr}, x::AbstractVector{Complex{Tr}}) where {Tr<:Real}
    work = similar(x)
    _apply_plan!(Val(:div), P, x, work)
end

Base.:*(P::ChebyshevPlan, x::AbstractVector) = (y = copy(x); lmul!(P, y); y)
Base.:\(P::ChebyshevPlan, x::AbstractVector) = (y = copy(x); ldiv!(P, y); y)

function LinearAlgebra.lmul!(P::ChebyshevLegendrePlan, x::AbstractVector)
    _apply_plan!(Val(:mul), P, x, similar(x))
end

function LinearAlgebra.ldiv!(P::ChebyshevLegendrePlan, x::AbstractVector)
    _apply_plan!(Val(:div), P, x, similar(x))
end

Base.:*(P::ChebyshevLegendrePlan, x::AbstractVector) = (y = copy(x); LinearAlgebra.lmul!(P, y); y)
Base.:\(P::ChebyshevLegendrePlan, x::AbstractVector) = (y = copy(x); LinearAlgebra.ldiv!(P, y); y)

# -----------------------------------------------------------------------------
# Chebyshev ↔ Legendre / transform plan application

@inline function _apply_plan!(::Val{:mul}, plan::FourierPlan{Tr}, buf::AbstractVector{Complex{Tr}},
                              work::AbstractVector{Complex{Tr}}) where {Tr}
    N = length(buf)
    @inbounds for pos in 1:N
        p = plan.perm[pos]
        work[p + 1] = buf[pos] / N
    end
    plan.fwd * work
    copyto!(buf, work)
    return buf
end

@inline function _apply_plan!(::Val{:div}, plan::FourierPlan{Tr}, buf::AbstractVector{Complex{Tr}},
                              work::AbstractVector{Complex{Tr}}) where {Tr}
    N = length(buf)
    copyto!(work, buf)
    plan.bwd * work
    @inbounds for pos in 1:N
        p = plan.perm[pos]
        buf[pos] = work[p + 1] * N
    end
    return buf
end

@inline function _apply_plan!(::Val{:mul}, plan::ChebyshevPlan{T}, buf::AbstractVector{T},
                              work::AbstractVector{T}) where {T<:Real}
    n = length(buf)
    @inbounds for pos in 1:n
        k = plan.idx[pos]
        work[k + 1] = buf[pos]
    end
    plan.fwd * work

    N = n - 1
    invN = one(T) / T(N)
    inv2N = invN / T(2)
    @inbounds begin
        work[1] *= inv2N
        for i in 2:(n - 1)
            work[i] *= invN
        end
        work[n] *= inv2N
    end

    copyto!(buf, work)
    return buf
end

@inline function _apply_plan!(::Val{:div}, plan::ChebyshevPlan{T}, buf::AbstractVector{T},
                              work::AbstractVector{T}) where {T<:Real}
    n = length(buf)
    @inbounds begin
        work[1] = buf[1]
        for i in 2:(n - 1)
            work[i] = buf[i] / T(2)
        end
        work[n] = buf[n]
    end

    plan.bwd * work

    @inbounds for pos in 1:n
        k = plan.idx[pos]
        buf[pos] = work[k + 1]
    end
    return buf
end


@inline function _apply_plan!(::Val{:mul}, plan::ChebyshevPlan{Tr}, buf::AbstractVector{Complex{Tr}},
                              work::AbstractVector{Complex{Tr}}) where {Tr<:Real}
    n = length(buf)
    tmp = Vector{Tr}(undef, n)

    @inbounds for pos in 1:n
        k = plan.idx[pos]
        tmp[k + 1] = real(buf[pos])
    end
    plan.fwd * tmp

    N = n - 1
    invN = one(Tr) / Tr(N)
    inv2N = invN / Tr(2)
    @inbounds begin
        tmp[1] *= inv2N
        for i in 2:(n - 1)
            tmp[i] *= invN
        end
        tmp[n] *= inv2N
        for pos in 1:n
            buf[pos] = Complex(tmp[pos], imag(buf[pos]))
        end
    end

    @inbounds for pos in 1:n
        k = plan.idx[pos]
        tmp[k + 1] = imag(buf[pos])
    end
    plan.fwd * tmp
    @inbounds begin
        tmp[1] *= inv2N
        for i in 2:(n - 1)
            tmp[i] *= invN
        end
        tmp[n] *= inv2N
        for pos in 1:n
            buf[pos] = Complex(real(buf[pos]), tmp[pos])
        end
    end
    return buf
end

@inline function _apply_plan!(::Val{:div}, plan::ChebyshevPlan{Tr}, buf::AbstractVector{Complex{Tr}},
                              work::AbstractVector{Complex{Tr}}) where {Tr<:Real}
    n = length(buf)
    tmp = Vector{Tr}(undef, n)

    @inbounds begin
        tmp[1] = real(buf[1])
        for i in 2:(n - 1)
            tmp[i] = real(buf[i]) / Tr(2)
        end
        tmp[n] = real(buf[n])
    end
    plan.bwd * tmp
    @inbounds for pos in 1:n
        k = plan.idx[pos]
        buf[pos] = Complex(tmp[k + 1], imag(buf[pos]))
    end

    @inbounds begin
        tmp[1] = imag(buf[1])
        for i in 2:(n - 1)
            tmp[i] = imag(buf[i]) / Tr(2)
        end
        tmp[n] = imag(buf[n])
    end
    plan.bwd * tmp
    @inbounds for pos in 1:n
        k = plan.idx[pos]
        buf[pos] = Complex(real(buf[pos]), tmp[k + 1])
    end
    return buf
end

@inline function _apply_plan!(::Val{:mul}, plan::ChebyshevLegendrePlan, buf::AbstractVector, work::AbstractVector)
    LinearAlgebra.lmul!(plan.fwd, buf)
    return buf
end

@inline function _apply_plan!(::Val{:div}, plan::ChebyshevLegendrePlan, buf::AbstractVector, work::AbstractVector)
    LinearAlgebra.lmul!(plan.bwd, buf)
    return buf
end

# -----------------------------------------------------------------------------
# Dirichlet Legendre basis kernels

raw"""Map Dirichlet-basis coefficients `u` to Legendre coefficients `c`.

`u[k]` corresponds to the coefficient of $\phi_k = L_k - L_{k+2}$.

The output `c` has length `length(u) + 2` and corresponds to the Legendre series

```math
\sum_j c_j L_j.
```
"""
function dirichlet_to_legendre!(c::AbstractVector{T}, u::AbstractVector{T}) where {T<:Number}
    m = length(u)
    length(c) == m + 2 || throw(DimensionMismatch("expected length(c) = length(u)+2 = $(m+2), got $(length(c))"))
    fill!(c, zero(T))

    @inbounds for k in 1:m
        c[k] += u[k]
        c[k + 2] -= u[k]
    end
    return c
end

"""Map Legendre coefficients `c` to Dirichlet-basis coefficients `u`.

This assumes `c` represents a function in the span of `{L_k - L_{k+2}}`.

The output `u` has length `length(c) - 2`.
"""
function legendre_to_dirichlet!(u::AbstractVector{T}, c::AbstractVector{T}) where {T<:Number}
    n = length(c)
    n >= 2 || throw(DimensionMismatch("expected length(c) ≥ 2, got $n"))
    length(u) == n - 2 || throw(DimensionMismatch("expected length(u) = length(c)-2 = $(n-2), got $(length(u))"))

    m = n - 2
    m == 0 && return u

    @inbounds begin
        u[1] = c[1]
        if m >= 2
            u[2] = c[2]
            for k in 3:m
                u[k] = c[k] + u[k - 2]
            end
        end
    end

    return u
end

raw"""Compute $f_k = (f, \phi_k)$ from Legendre coefficients of $f$.

If

```math
f(x) = \sum_{j=0}^{N} c_j L_j(x),\qquad \phi_k = L_k - L_{k+2},
```

then by Legendre orthogonality

```math
(f, \phi_k) = \frac{2}{2k+1}c_k - \frac{2}{2k+5}c_{k+2}.
```

The output vector has length `length(c) - 2`.
"""
function legendre_dirichlet_rhs!(out::AbstractVector{T}, c::AbstractVector{T}) where {T<:Number}
    n = length(c)
    n >= 2 || throw(DimensionMismatch("expected length(c) ≥ 2, got $n"))
    length(out) == n - 2 || throw(DimensionMismatch("expected length(out) = length(c)-2 = $(n-2), got $(length(out))"))

    m = n - 2
    @inbounds for i in 1:m
        out[i] = (T(2) / T(2i - 1)) * c[i] - (T(2) / T(2i + 3)) * c[i + 2]
    end
    return out
end

# -----------------------------------------------------------------------------
# Hierarchization / dehierarchization sweeps (size-based)

function chebyshev_hierarchize!(coeffs::AbstractVector)
    hi = length(coeffs) - 1
    while hi > 1
        lo = (hi >>> 1) + 1
        @inbounds for k in hi:-1:lo
            coeffs[(hi - k) + 1] += coeffs[k + 1]
        end
        hi >>>= 1
    end
    return coeffs
end

function chebyshev_dehierarchize!(coeffs::AbstractVector)
    hi = 2
    N = length(coeffs) - 1
    while hi <= N
        lo = (hi >>> 1) + 1
        @inbounds for k in lo:hi
            coeffs[(hi - k) + 1] -= coeffs[k + 1]
        end
        hi <<= 1
    end
    return coeffs
end

function chebyshev_dehierarchize_transpose!(coeffs::AbstractVector)
    hi = length(coeffs) - 1
    while hi > 1
        lo = (hi >>> 1) + 1
        @inbounds for k in hi:-1:lo
            coeffs[k + 1] -= coeffs[(hi - k) + 1]
        end
        hi >>>= 1
    end
    return coeffs
end

function chebyshev_hierarchize_transpose!(coeffs::AbstractVector)
    hi = 2
    N = length(coeffs) - 1
    while hi <= N
        lo = (hi >>> 1) + 1
        @inbounds for k in lo:hi
            coeffs[k + 1] += coeffs[(hi - k) + 1]
        end
        hi <<= 1
    end
    return coeffs
end

function fourier_hierarchize!(coeffs::AbstractVector)
    hi = length(coeffs)
    while hi > 1
        half = hi >>> 1
        lo = half + 1
        @inbounds for k in hi:-1:lo
            coeffs[k - half] += coeffs[k]
        end
        hi >>>= 1
    end
    return coeffs
end

function fourier_dehierarchize!(coeffs::AbstractVector)
    hi = 2
    N = length(coeffs)
    while hi <= N
        half = hi >>> 1
        lo = half + 1
        @inbounds for k in lo:hi
            coeffs[k - half] -= coeffs[k]
        end
        hi <<= 1
    end
    return coeffs
end

function fourier_dehierarchize_transpose!(coeffs::AbstractVector)
    hi = length(coeffs)
    while hi > 1
        half = hi >>> 1
        lo = half + 1
        @inbounds for k in hi:-1:lo
            coeffs[k] -= coeffs[k - half]
        end
        hi >>>= 1
    end
    return coeffs
end

function fourier_hierarchize_transpose!(coeffs::AbstractVector)
    hi = 2
    N = length(coeffs)
    while hi <= N
        half = hi >>> 1
        lo = half + 1
        @inbounds for k in lo:hi
            coeffs[k] += coeffs[k - half]
        end
        hi <<= 1
    end
    return coeffs
end

"""Shared dyadic parent lookups up to refinement index `rmax`."""
struct DyadicHierarchyShared{Ti<:Integer}
    pL::Vector{Vector{Ti}}
    pR::Vector{Vector{Ti}}
end

function DyadicHierarchyShared(nodes::DyadicNodes, rmax::Int, ::Type{Ti}=Int) where {Ti<:Integer}
    rmax < 0 && throw(ArgumentError("rmax must be ≥ 0"))

    offset = totalsize(nodes, 0)
    pL = Vector{Vector{Ti}}(undef, rmax + 1)
    pR = Vector{Vector{Ti}}(undef, rmax + 1)
    pL[1] = Ti[]
    pR[1] = Ti[]

    @inbounds for r in 1:rmax
        m = 1 << (r - 1)
        pl = Vector{Ti}(undef, m)
        pr = Vector{Ti}(undef, m)

        pos = Vector{Ti}(undef, m + 1)
        parent_r = r - 1
        N = 1 << parent_r

        for k in 0:m
            if k == 0
                pos[k + 1] = Ti(offset == 2 ? 1 : 0)
            elseif k == N
                pos[k + 1] = Ti(offset == 2 ? 2 : 0)
            else
                tz = trailing_zeros(k)
                rmin = parent_r - tz
                start = (1 << (rmin - 1)) + offset
                kk = k >> tz
                q = (kk - 1) >>> 1
                pos[k + 1] = Ti(start + q)
            end
        end

        for q in 0:(m - 1)
            pl[q + 1] = pos[q + 1]
            pr[q + 1] = pos[q + 2]
        end

        pL[r + 1] = pl
        pR[r + 1] = pr
    end

    return DyadicHierarchyShared{Ti}(pL, pR)
end

struct HierarchyPlan{Axis,Aux}
    axis::Axis
    aux::Aux
end

@inline _dyadic_aux(shared::DyadicHierarchyShared, r::Int) = (pL = shared.pL, pR = shared.pR, r = r)

function _dyadic_hierarchize!(v::AbstractVector, nodes::DyadicNodes{<:LevelOrder}, aux)
    offset = totalsize(nodes, 0)
    z = zero(eltype(v))
    @inbounds for rr in aux.r:-1:1
        pL = aux.pL[rr + 1]
        pR = aux.pR[rr + 1]
        m = length(pL)
        start = m + offset
        for j in 1:m
            pos = start + (j - 1)
            l = pL[j] == 0 ? z : v[pL[j]]
            r = pR[j] == 0 ? z : v[pR[j]]
            v[pos] -= (l + r) / 2
        end
    end
    return v
end

function _dyadic_dehierarchize!(v::AbstractVector, nodes::DyadicNodes{<:LevelOrder}, aux)
    offset = totalsize(nodes, 0)
    z = zero(eltype(v))
    @inbounds for rr in 1:aux.r
        pL = aux.pL[rr + 1]
        pR = aux.pR[rr + 1]
        m = length(pL)
        start = m + offset
        for j in 1:m
            pos = start + (j - 1)
            l = pL[j] == 0 ? z : v[pL[j]]
            r = pR[j] == 0 ? z : v[pR[j]]
            v[pos] += (l + r) / 2
        end
    end
    return v
end

function _dyadic_dehierarchize_transpose!(v::AbstractVector, nodes::DyadicNodes{<:LevelOrder}, aux)
    @inbounds for rr in aux.r:-1:1
        pL = aux.pL[rr + 1]
        pR = aux.pR[rr + 1]
        m = length(pL)
        start = m + totalsize(nodes, 0)
        for j in 1:m
            pos = start + (j - 1)
            t = v[pos] / 2
            pL[j] == 0 || (v[pL[j]] += t)
            pR[j] == 0 || (v[pR[j]] += t)
        end
    end
    return v
end

function _dyadic_hierarchize_transpose!(v::AbstractVector, nodes::DyadicNodes{<:LevelOrder}, aux)
    @inbounds for rr in 1:aux.r
        pL = aux.pL[rr + 1]
        pR = aux.pR[rr + 1]
        m = length(pL)
        start = m + totalsize(nodes, 0)
        for j in 1:m
            pos = start + (j - 1)
            t = v[pos] / 2
            pL[j] == 0 || (v[pL[j]] -= t)
            pR[j] == 0 || (v[pR[j]] -= t)
        end
    end
    return v
end

function LinearAlgebra.lmul!(P::HierarchyPlan, x::AbstractVector)
    _apply_plan!(Val(:mul), P, x, similar(x))
end

function LinearAlgebra.ldiv!(P::HierarchyPlan, x::AbstractVector)
    _apply_plan!(Val(:div), P, x, similar(x))
end

Base.:*(P::HierarchyPlan, x::AbstractVector) = (y = copy(x); lmul!(P, y); y)
Base.:\(P::HierarchyPlan, x::AbstractVector) = (y = copy(x); ldiv!(P, y); y)

# -----------------------------------------------------------------------------
# Plan builders

function make_plan_entry(::LineTransform, axis::FourierEquispacedNodes, n::Integer, r::Integer, ::Type{T}, shared) where {T<:Number}
    return FourierPlan(axis, n, T)
end

function make_plan_entry(::LineTransform, axis::ChebyshevGaussLobattoNodes, n::Integer, r::Integer, ::Type{T}, shared) where {T<:Number}
    return ChebyshevPlan(axis, n, T)
end

function make_plan_entry(::LineChebyshevLegendre, axis::ChebyshevGaussLobattoNodes, n::Integer, r::Integer, ::Type{T}, shared) where {T<:Real}
    return ChebyshevLegendrePlan(n, T)
end

function make_plan_entry(::LineHierarchize, axis::ChebyshevGaussLobattoNodes{<:Any,EndpointMask{0x3}}, n::Integer, r::Integer, ::Type{T}, shared) where {T}
    return HierarchyPlan(axis, nothing)
end
make_plan_entry(::LineDehierarchize, axis::ChebyshevGaussLobattoNodes{<:Any,EndpointMask{0x3}}, n::Integer, r::Integer, ::Type{T}, shared) where {T} =
    HierarchyPlan(axis, nothing)
make_plan_entry(::LineHierarchizeTranspose, axis::ChebyshevGaussLobattoNodes{<:Any,EndpointMask{0x3}}, n::Integer, r::Integer, ::Type{T}, shared) where {T} =
    HierarchyPlan(axis, nothing)
make_plan_entry(::LineDehierarchizeTranspose, axis::ChebyshevGaussLobattoNodes{<:Any,EndpointMask{0x3}}, n::Integer, r::Integer, ::Type{T}, shared) where {T} =
    HierarchyPlan(axis, nothing)

function make_plan_entry(op::Union{LineHierarchize,LineDehierarchize,LineHierarchizeTranspose,LineDehierarchizeTranspose},
                         ::ChebyshevGaussLobattoNodes{<:Any,EM}, n::Integer, r::Integer, ::Type{T}, shared) where {T,EM<:AbstractEndpointMode}
    throw(ArgumentError("$(typeof(op)) is only implemented for ChebyshevGaussLobattoNodes with both endpoints"))
end

function make_plan_entry(::LineHierarchize, axis::FourierEquispacedNodes, n::Integer, r::Integer, ::Type{T}, shared) where {T}
    return HierarchyPlan(axis, nothing)
end
make_plan_entry(::LineDehierarchize, axis::FourierEquispacedNodes, n::Integer, r::Integer, ::Type{T}, shared) where {T} =
    HierarchyPlan(axis, nothing)
make_plan_entry(::LineHierarchizeTranspose, axis::FourierEquispacedNodes, n::Integer, r::Integer, ::Type{T}, shared) where {T} =
    HierarchyPlan(axis, nothing)
make_plan_entry(::LineDehierarchizeTranspose, axis::FourierEquispacedNodes, n::Integer, r::Integer, ::Type{T}, shared) where {T} =
    HierarchyPlan(axis, nothing)

make_plan_shared(::LineHierarchize, axis::DyadicNodes{<:LevelOrder}, rmax::Integer, ::Type{T}) where {T} =
    DyadicHierarchyShared(axis, Int(rmax), Int)
make_plan_shared(::LineDehierarchize, axis::DyadicNodes{<:LevelOrder}, rmax::Integer, ::Type{T}) where {T} =
    DyadicHierarchyShared(axis, Int(rmax), Int)
make_plan_shared(::LineHierarchizeTranspose, axis::DyadicNodes{<:LevelOrder}, rmax::Integer, ::Type{T}) where {T} =
    DyadicHierarchyShared(axis, Int(rmax), Int)
make_plan_shared(::LineDehierarchizeTranspose, axis::DyadicNodes{<:LevelOrder}, rmax::Integer, ::Type{T}) where {T} =
    DyadicHierarchyShared(axis, Int(rmax), Int)

function make_plan_entry(::LineHierarchize, axis::DyadicNodes{<:LevelOrder}, n::Integer, r::Integer, ::Type{T}, shared::DyadicHierarchyShared) where {T}
    return HierarchyPlan(axis, _dyadic_aux(shared, Int(r)))
end

make_plan_entry(::LineDehierarchize, axis::DyadicNodes{<:LevelOrder}, n::Integer, r::Integer, ::Type{T}, shared::DyadicHierarchyShared) where {T} =
    HierarchyPlan(axis, _dyadic_aux(shared, Int(r)))
make_plan_entry(::LineHierarchizeTranspose, axis::DyadicNodes{<:LevelOrder}, n::Integer, r::Integer, ::Type{T}, shared::DyadicHierarchyShared) where {T} =
    HierarchyPlan(axis, _dyadic_aux(shared, Int(r)))
make_plan_entry(::LineDehierarchizeTranspose, axis::DyadicNodes{<:LevelOrder}, n::Integer, r::Integer, ::Type{T}, shared::DyadicHierarchyShared) where {T} =
    HierarchyPlan(axis, _dyadic_aux(shared, Int(r)))

# -----------------------------------------------------------------------------
# Plan application for hierarchy family

@inline function _apply_plan!(::Val{:mul}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:ChebyshevGaussLobattoNodes}
    chebyshev_hierarchize!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:div}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:ChebyshevGaussLobattoNodes}
    chebyshev_dehierarchize!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:tmul}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:ChebyshevGaussLobattoNodes}
    chebyshev_hierarchize_transpose!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:tdiv}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:ChebyshevGaussLobattoNodes}
    chebyshev_dehierarchize_transpose!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:mul}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:FourierEquispacedNodes}
    fourier_hierarchize!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:div}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:FourierEquispacedNodes}
    fourier_dehierarchize!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:tmul}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:FourierEquispacedNodes}
    fourier_hierarchize_transpose!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:tdiv}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:FourierEquispacedNodes}
    fourier_dehierarchize_transpose!(buf)
    return buf
end

@inline function _apply_plan!(::Val{:mul}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:DyadicNodes{<:LevelOrder}}
    _dyadic_hierarchize!(buf, plan.axis, plan.aux)
    return buf
end

@inline function _apply_plan!(::Val{:div}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:DyadicNodes{<:LevelOrder}}
    _dyadic_dehierarchize!(buf, plan.axis, plan.aux)
    return buf
end

@inline function _apply_plan!(::Val{:tmul}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:DyadicNodes{<:LevelOrder}}
    _dyadic_hierarchize_transpose!(buf, plan.axis, plan.aux)
    return buf
end

@inline function _apply_plan!(::Val{:tdiv}, plan::HierarchyPlan{Axis}, buf::AbstractVector, work::AbstractVector) where {Axis<:DyadicNodes{<:LevelOrder}}
    _dyadic_dehierarchize_transpose!(buf, plan.axis, plan.aux)
    return buf
end
