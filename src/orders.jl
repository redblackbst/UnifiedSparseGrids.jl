"""Ordering of 1D nodes inside each level.

The sparse grid traversal and transform code in this package assumes that
nodal samples are stored in a *nested* 1D order that is friendly for radix-2
fast transforms.
"""
abstract type AbstractNodeOrder end

"""Natural/intuitive order (typically ascending coordinate order)."""
struct NaturalOrder <: AbstractNodeOrder end

"""Level-order / even–odd split order.

This is the canonical storage order for nested 1D node sets in this package.

It is the order induced by recursively applying Perera–Liu's *even–odd*
permutation matrix ``P_n`` (group even indices first, then odd indices),
which appears naturally in radix-2 DIT/DIF factorizations of FFT/DCT-style
transforms:

``P_n x = [x_0, x_2, x_4, …, x_{n-1},  x_1, x_3, x_5, …]``.

Recursively applying this even–odd split yields the familiar *bit-reversal*
ordering on power-of-two grids. For nested sparse grid rules this is also
often called *hierarchical level order* (points grouped by Δ-level, with an
even–odd split inside each Δ-level).

Example (Chebyshev–Gauss–Lobatto, ``N=8`` intervals, indices ``k=0..8``)::

    [0, 8, 4, 2, 6, 1, 3, 5, 7]

The endpoints ``0`` and ``N`` come first (Δℓ=0), then the midpoint (Δℓ=1),
then the newly introduced odd points at each finer level.

!!! note
    For some node families (notably Chebyshev–Gauss–Lobatto / Clenshaw–Curtis),
    this package fixes a **within-level left-to-right** order (odd indices
    increasing) rather than a strict bit-reversal within each Δ-level. This
    keeps the global order hierarchical while making certain DCT recursions
    permutation-free.
"""
struct LevelOrder <: AbstractNodeOrder end



"""
    bitrev(j, m) -> Int

Reverse the lowest `m` bits of a nonnegative integer `j`.

This is the standard permutation used by radix-2 FFTs when mapping between
in-order and bit-reversed indices.
"""
function bitrev(j::Integer, m::Integer)::Int
    if j < 0
        throw(ArgumentError("bitrev requires j ≥ 0, got $j"))
    end
    if m < 0
        throw(ArgumentError("bitrev requires m ≥ 0, got $m"))
    end
    x = UInt(j)
    r = UInt(0)
    for _ in 1:m
        r = (r << 1) | (x & UInt(1))
        x >>= 1
    end
    return Int(r)
end

"""
    bitrevperm(N) -> Vector{Int}

Return the radix-2 bit-reversal permutation for `N` points.

`N` must be a power of two. The returned vector `p` has length `N` and
satisfies `p[i+1] == bitrev(i, log2(N))` for `i = 0:(N-1)`.
"""
function bitrevperm(N::Integer)::Vector{Int}
    if N <= 0
        throw(ArgumentError("bitrevperm requires N > 0, got $N"))
    end
    if (N & (N - 1)) != 0
        throw(ArgumentError("bitrevperm requires N to be a power of two, got $N"))
    end
    m = trailing_zeros(UInt(N))
    p = Vector{Int}(undef, N)
    for i in 0:(N - 1)
        p[i + 1] = bitrev(i, m)
    end
    return p
end
