
using Test
using Random
using UnifiedSparseGrids

function _tensor_points_to_scattered(P::TensorProductPoints)
    D = ndims(P)
    lengths = ntuple(d -> length(P.pts[d]), D)
    X = Matrix{Float64}(undef, D, prod(lengths))
    lin = LinearIndices(CartesianIndices(lengths))
    for Ipt in CartesianIndices(lengths)
        j = lin[Ipt]
        @inbounds for d in 1:D
            X[d, j] = P.pts[d][Ipt[d]]
        end
    end
    return ScatteredPoints(X)
end

@testset "Evaluation API and point-set contract" begin
    x1 = collect(range(-1.0, 1.0; length=3))
    x2 = collect(range(-1.0, 1.0; length=4))
    P = TensorProductPoints((x1, x2))
    @test ndims(P) == 2
    @test length(P) == 12

    X = randn(2, 7)
    S = ScatteredPoints(X)
    @test ndims(S) == 2
    @test length(S) == 7
end

@testset "Backend selection and parity" begin
    Random.seed!(0xC0FFEE)
    D = 2
    L = 3
    cap = (3, 2)
    grid = SparseGrid(SparseGridSpec(ntuple(_ -> ChebyshevGaussLobattoNodes(), D), SmolyakIndexSet(Val(D), L; cap=cap)))
    P = TensorProductPoints((range(-1, 1; length=9), range(-1, 1; length=7)))
    coeffs = randn(length(grid))

    plan_auto = plan_evaluate(grid, ChebyshevTBasis(), P, Float64; backend=:auto)
    @test plan_auto.backend isa UnifiedSparseGrids.UnidirectionalBackend

    plan_uni = plan_evaluate(grid, ChebyshevTBasis(), P, Float64; backend=:unidirectional)
    plan_naive = plan_evaluate(grid, ChebyshevTBasis(), P, Float64; backend=:naive)
    @test evaluate(plan_uni, coeffs) ≈ evaluate(plan_naive, coeffs)

    S = ScatteredPoints(rand(D, 10))
    plan_auto2 = plan_evaluate(grid, ChebyshevTBasis(), S, Float64; backend=:auto)
    @test plan_auto2.backend isa UnifiedSparseGrids.NaiveBackend
end

@testset "Tensor-product vs scattered evaluation" begin
    Random.seed!(0)
    for D in 2:3
        L = 3
        cap = ntuple(d -> max(L - (d - 1), 0), D)
        axes = ntuple(_ -> ChebyshevGaussLobattoNodes(), D)
        grid = SparseGrid(SparseGridSpec(axes, SmolyakIndexSet(Val(D), L; cap=cap)))
        basis = ChebyshevTBasis()
        coeffs = randn(length(grid))
        P = D == 2 ? TensorProductPoints((collect(range(-1.0, 1.0; length=7)), collect(range(-1.0, 1.0; length=19)))) :
                     TensorProductPoints((collect(range(-1.0, 1.0; length=7)), collect(range(-1.0, 1.0; length=19)), collect(range(-1.0, 1.0; length=5))))
        S = _tensor_points_to_scattered(P)
        y_tp = evaluate(plan_evaluate(grid, basis, P, Float64; backend=:unidirectional), coeffs)
        y_sc = evaluate(plan_evaluate(grid, basis, S, Float64; backend=:naive), coeffs)
        @test y_tp ≈ y_sc
    end
end

@testset "Basis-aware sparse grid evaluation sanity" begin
    Random.seed!(1)
    for D in 2:3
        L = 3
        cap = ntuple(d -> max(L - (d - 1), 0), D)
        grid = SparseGrid(SparseGridSpec(ntuple(_ -> DyadicNodes(), D), SmolyakIndexSet(Val(D), L; cap=cap)))
        bases = D == 2 ? (HatBasis(), PiecewisePolynomialBasis(3)) : (HatBasis(), HatBasis(), PiecewisePolynomialBasis(3))
        pts = ScatteredPoints(rand(D, 20))
        coeffs = randn(length(grid))

        plan_auto = plan_evaluate(grid, bases, pts, Float64; backend=:auto)
        @test plan_auto.backend isa UnifiedSparseGrids.PseudorecursiveBackend

        plan_pr = plan_evaluate(grid, bases, pts, Float64; backend=:pseudorecursive)
        plan_nv = plan_evaluate(grid, bases, pts, Float64; backend=:naive)
        @test evaluate(plan_pr, coeffs) ≈ evaluate(plan_nv, coeffs) atol=1e-12 rtol=1e-12
    end
end
