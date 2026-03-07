"""Balder–Zenger (1996) hat-basis Helmholtz example.

This script solves the hat-basis Helmholtz system

    (K + c*M) u = b,

on a dyadic **sparse grid** (Smolyak index set), where the 1D hat mass operator
uses the Balder–Zenger Eq. (15) decomposition and the 1D stiffness is diagonal
(Balder–Zenger Eq. (9)-(10)).

Following Balder–Zenger (1996), we approximate the exact solution by solving
on a *finer sparse grid* (rather than a full tensor grid, which becomes
intractable quickly).

RHS is constructed as

    b = c * (M⊗⋯⊗M) * w,

where `w` is the tensor product of the refinement-1 interior hat (the center hat in 1D).

We report a **discrete L2(M) error ratio** against the reference solve:

    ||u_sparse - u_ref||_M / ||u_ref||_M,

where `||v||_M := sqrt(v' (M⊗⋯⊗M) v)`.

Run:

    julia --project=. examples/balder_zenger_1996_helmholtz.jl --d=2 --level=7 --c=10.0
"""

using UnifiedSparseGrids
using StaticArrays
using LinearAlgebra
using Krylov

@inline function _parse_kv_args(args)
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

"""Construct `w`: tensor-product of the refinement-1 interior hat.

In dyadic LevelOrder with endpoints, the first three 1D points are `[0, 1, 1/2]`.
The refinement-1 interior hat corresponds to the 1D position `3`.
"""
function level1_tensor_hat(grid::SparseGrid{<:SparseGridSpec{D}}) where {D}
    w = zeros(Float64, length(grid))
    target = SVector{D,Int}(ntuple(_ -> 3, Val(D)))
    it = traverse(grid)
    @inbounds for (i, c) in enumerate(it)
        if SVector{D,Int}(c) == target
            w[i] = 1.0
            return w
        end
    end
    throw(ArgumentError("refinement-1 interior hat coordinate $target not present in grid"))
end

"""Build `M⊗⋯⊗M` on `grid` as a `TensorSumMatVec` with a single term."""
function mass_tensor_op(grid::SparseGrid{<:SparseGridSpec{D}}; T::Type{<:Real}=Float64) where {D}
    rmax = refinement_caps(grid)
    axes = grid.spec.axes
    mass_ops = ntuple(d -> HatMassLineOp(axes[d], Int(rmax[d]); T=T), Val(D))
    term = WeightedTensorTerm(one(T), UpDownTensorOp(Tuple(mass_ops); omit_dim=0))
    return TensorSumMatVec(grid, (term,), T)
end

"""Compute `||u||_M` with `M⊗⋯⊗M` as the discrete L2 norm."""
function mass_norm(grid::SparseGrid, u::AbstractVector{T}) where {T<:Real}
    M = mass_tensor_op(grid; T=T)
    Mu = similar(u)
    mul!(Mu, M, u)
    return sqrt(dot(u, Mu))
end

"""Diagonal of the hat Helmholtz operator on `grid` (for Jacobi)."""
function helmholtz_hat_diag(grid::SparseGrid{<:SparseGridSpec{D}}; c::Real, T::Type{<:Real}=Float64) where {D}
    rmax = refinement_caps(grid)
    axes = grid.spec.axes
    mass_ops = ntuple(d -> HatMassLineOp(axes[d], Int(rmax[d]); T=T), Val(D))
    stiff_ops = ntuple(d -> HatStiffnessDiagLineOp(axes[d], Int(rmax[d]); T=T), Val(D))

    diagM = ntuple(d -> begin
        h = mass_ops[d].hdiag[Int(rmax[d]) + 1]
        (T(2) / T(3)) .* h
    end, Val(D))
    diagS = ntuple(d -> stiff_ops[d].diags[Int(rmax[d]) + 1], Val(D))

    diag = Vector{T}(undef, length(grid))
    it = traverse(grid)
    cT = T(c)
    @inbounds for (i, coords) in enumerate(it)
        dm = ntuple(d -> diagM[d][coords[d]], Val(D))
        prodM = one(T)
        for d in 1:D
            prodM *= dm[d]
        end
        v = cT * prodM
        for s in 1:D
            v += diagS[s][coords[s]] * (prodM / dm[s])
        end
        diag[i] = v
    end
    return diag
end

"""Build the sparse grid tensor-sum operator for `K + c*M` on `grid`."""
function helmholtz_hat_tensor_sum(grid::SparseGrid{<:SparseGridSpec{D}}; c::Real, T::Type{<:Real}=Float64) where {D}
    rmax = refinement_caps(grid)
    axes = grid.spec.axes
    mass_ops = ntuple(d -> HatMassLineOp(axes[d], Int(rmax[d]); T=T), Val(D))
    stiff_ops = ntuple(d -> HatStiffnessDiagLineOp(axes[d], Int(rmax[d]); T=T), Val(D))

    mass_term = WeightedTensorTerm(T(c), UpDownTensorOp(Tuple(mass_ops); omit_dim=0))
    stiff_terms = ntuple(s -> begin
        ops_s = ntuple(d -> (d == s ? stiff_ops[d] : mass_ops[d]), Val(D))
        WeightedTensorTerm(one(T), UpDownTensorOp(Tuple(ops_s); omit_dim=0))
    end, Val(D))

    terms = T(c) == zero(T) ? stiff_terms : (mass_term, stiff_terms...)
    return TensorSumMatVec(grid, terms, T)
end

"""Solve (K + cM)u=b on `grid` with CG + Jacobi."""
function solve_hat_helmholtz(grid::SparseGrid, b::AbstractVector{T};
                             c::Real,
                             rtol::Real=1e-10,
                             atol::Real=1e-12,
                             itmax::Int=0,
                             verbose::Int=0) where {T<:Real}
    A = helmholtz_hat_tensor_sum(grid; c=c, T=T)
    diag = helmholtz_hat_diag(grid; c=c, T=T)
    M = jacobi_precond(diag)

    Aop = as_linear_operator(A; symmetric=true, hermitian=true)
    x, stats = Krylov.cg(Aop, b; M=M, ldiv=false, rtol=T(rtol), atol=T(atol), itmax=itmax, verbose=verbose)
    return x, stats
end

function run_case(; d::Int=2, level::Int=7, c::Float64=10.0, level_ref::Int=0, verbose::Int=0)
    level_ref == 0 && (level_ref = level + 2)

    cap = fill(max(level, level_ref), d)

    # Dyadic axes in LevelOrder (required by dyadic hierarchization).
    axes = ntuple(_ -> DyadicNodes(LevelOrder()), d)

    I_sparse = SmolyakIndexSet(d, level; cap=cap)
    I_ref = SmolyakIndexSet(d, level_ref; cap=cap)

    grid_sparse = SparseGrid(SparseGridSpec(axes, I_sparse))
    grid_ref = SparseGrid(SparseGridSpec(axes, I_ref))

    println("d=$d level=$level level_ref=$level_ref c=$c")
    println("  sparse n=$(length(grid_sparse)), ref n=$(length(grid_ref))")

    # RHS: b = c * (M⊗⋯⊗M) * w
    wS = level1_tensor_hat(grid_sparse)
    MS = mass_tensor_op(grid_sparse; T=Float64)
    bS = similar(wS)
    mul!(bS, MS, wS)
    @. bS = c * bS

    wR = level1_tensor_hat(grid_ref)
    MR = mass_tensor_op(grid_ref; T=Float64)
    bR = similar(wR)
    mul!(bR, MR, wR)
    @. bR = c * bR

    uS, statsS = solve_hat_helmholtz(grid_sparse, bS; c=c, rtol=1e-10, atol=1e-12, itmax=0, verbose=verbose)
    uR, statsR = solve_hat_helmholtz(grid_ref, bR; c=c, rtol=1e-10, atol=1e-12, itmax=0, verbose=verbose)

    println("  sparse: iters=$(statsS.niter) solved=$(statsS.solved)")
    println("  ref:    iters=$(statsR.niter) solved=$(statsR.solved)")

    # Prolong sparse solution into the reference grid and report discrete L2 error ratio.
    plan = TransferPlan(grid_sparse, grid_ref)
    uS_ref = zeros(Float64, length(grid_ref))
    embed!(uS_ref, uS, plan)

    err = uS_ref .- uR
    rel = mass_norm(grid_ref, err) / mass_norm(grid_ref, uR)
    println("  rel L2(M) error vs ref: $rel")

    return (u_sparse=uS, u_ref=uR, rel_L2=rel, stats_sparse=statsS, stats_ref=statsR)
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = _parse_kv_args(ARGS)
    d = parse(Int, get(kv, "d", "2"))
    level = parse(Int, get(kv, "level", "7"))
    level_ref = parse(Int, get(kv, "level_ref", "0"))
    c = parse(Float64, get(kv, "c", "10.0"))
    verbose = parse(Int, get(kv, "verbose", "0"))
    run_case(; d, level, level_ref, c, verbose)
end
