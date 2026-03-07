# -----------------------------------------------------------------------------
# Combination technique (Bungartz–Griebel)

"""A single full-grid subproblem in the sparse grid combination technique.

Fields:

* `cap`: full-tensor refinement-cap vector `L` (0-based), meaning the tensor grid includes
  all subspaces `W_ℓ` with `ℓ ≤ L` componentwise.
* `weight`: integer combination weight.
"""
struct CombinationSubproblem{D}
    cap::SVector{D,Int}
    weight::Int
end

function _enumerate_fixed_sum(::Val{D}, s::Int, cap::SVector{D,Int}) where {D}
    s < 0 && return SVector{D,Int}[]

    out = SVector{D,Int}[]
    ℓ = MVector{D,Int}(ntuple(_ -> 0, Val(D)))

    function rec(dim::Int, rem::Int)
        if dim == D
            t = rem
            0 <= t <= cap[dim] || return
            ℓ[dim] = t
            push!(out, SVector{D,Int}(ℓ))
            return
        end

        max_t = min(cap[dim], rem)
        for t in 0:max_t
            ℓ[dim] = t
            rec(dim + 1, rem - t)
        end
        return
    end

    rec(1, s)

    # Deterministic colex ordering within a fixed |ℓ|₁ layer.
    sort!(out; by=ℓ -> Tuple(reverse(ℓ)))
    return out
end

"""Enumerate full-grid subproblems for the sparse grid combination technique.

For dimension `d` and level parameter `n`, this returns the collection of
full-tensor refinement-cap vectors `L` and their integer weights.

The combination layers are grouped by `|L|₁`:

* `|L|₁ = n + d - 1` has weight `+binomial(d-1, 0)`
* `|L|₁ = n + d - 2` has weight `-binomial(d-1, 1)`
* …
* `|L|₁ = n` has weight `(-1)^(d-1) * binomial(d-1, d-1)`

Keyword arguments:

* `cap`: optional per-dimension upper bound on `L` (isotropic `Integer` or
  `NTuple{d}` / `SVector{d}` cap).

Notes:

This function uses the package's **0-based** level convention.
"""
function each_combination_subproblem(d::Integer, n::Integer; cap=nothing)
    d >= 1 || throw(ArgumentError("d must satisfy d ≥ 1, got d=$d"))
    return each_combination_subproblem(Val(Int(d)), Int(n); cap=cap)
end

function each_combination_subproblem(::Val{D}, n::Int; cap=nothing) where {D}
    n < 0 && throw(ArgumentError("n must satisfy n ≥ 0, got n=$n"))

    cap_vec = cap === nothing ? nothing : _cap_convert(Val(D), Int, cap)

    subs = CombinationSubproblem{D}[]

    for q in 0:(D - 1)
        s = n + (D - 1) - q
        w = (isodd(q) ? -1 : 1) * binomial(D - 1, q)

        c = cap_vec === nothing ? _cap_default(Val(D), Int(s)) : min.(cap_vec, s)
        levels = _enumerate_fixed_sum(Val(D), s, c)
        for ℓ in levels
            push!(subs, CombinationSubproblem{D}(ℓ, Int(w)))
        end
    end

    return subs
end
