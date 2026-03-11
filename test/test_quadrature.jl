using Test
using Base.Threads: nthreads
using StaticArrays
using UnifiedSparseGrids

# local helpers
_rule_apply(rule::QuadratureRule, f) = sum(qweights(rule) .* map(f, qpoints(rule)))

function _manual_tensor_inclusion_exclusion(f, families::NTuple{D,AbstractQuadratureFamily}, r::SVector{D,Int}) where {D}
    acc = 0.0 + 0.0im
    for mask in 0:((1 << D) - 1)
        rr = ntuple(d -> r[d] - ((mask >> (d - 1)) & 0x1), D)
        any(<(0), rr) && continue
        rules = ntuple(d -> qrule(families[d], rr[d]), D)
        pts = ntuple(d -> qpoints(rules[d]), D)
        wts = ntuple(d -> qweights(rules[d]), D)
        val = zero(acc)
        for J in CartesianIndices(ntuple(d -> length(pts[d]), D))
            x = SVector{D,Float64}(ntuple(d -> pts[d][J[d]], D))
            w = one(Float64)
            for d in 1:D
                w *= wts[d][J[d]]
            end
            val += w * f(x)
        end
        if isodd(count_ones(mask))
            acc -= val
        else
            acc += val
        end
    end
    return acc
end

@testset "Quadrature rules and adaptive integration" begin
    @testset "Gauss-Legendre exactness metadata and moments" begin
        Q = GaussLegendreQuadrature()
        r = 2
        rule = qrule(Q, r)
        @test qdegree(Q, r) == 2qsize(Q, r) - 1
        for k in 0:(qdegree(Q, r))
            val = _rule_apply(rule, x -> x^k)
            exact = isodd(k) ? 0.0 : 2 / (k + 1)
            @test isapprox(val, exact; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "Nested qdiffrule telescoping" begin
        Q = ClenshawCurtisQuadrature()
        f(x) = exp(x) + x^3 - 2x
        for r in 0:4
            lhs = sum(_rule_apply(qdiffrule(Q, j), f) for j in 0:r)
            rhs = _rule_apply(qrule(Q, r), f)
            @test isapprox(lhs, rhs; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "Nested direct-difference equals inclusion-exclusion on small tensor case" begin
        Q = ClenshawCurtisQuadrature()
        fams = (Q, Q)
        f(x) = (x[1]^2 + 2x[1]*x[2] + x[2]^3) + 0.25im*(x[1] - x[2]^2)
        r = SVector(2, 1)
        direct, _, _ = delta_contribution(f, fams, r)
        manual = _manual_tensor_inclusion_exclusion(f, fams, r)
        @test isapprox(direct, manual; atol=1e-12, rtol=1e-12)
    end

    @testset "Adaptive frontier growth prefers active dimension" begin
        Q = ClenshawCurtisQuadrature()
        env = FullTensorIndexSet(2, 2)
        f(x) = x[1]^2
        _, state = integrate_adaptive(f, (Q, Q), env; indicator=:absdelta, maxterms=2, atol=0.0, rtol=0.0)
        @test length(state.accepted) == 2
        @test state.accepted[1] == SVector(0, 0)
        @test state.accepted[2] == SVector(1, 0)
    end

    @testset "Complex-valued integrands" begin
        Q = GaussLegendreQuadrature()
        env = FullTensorIndexSet(1, 3)
        f(x) = x[1] + im * x[1]^2
        val, _ = integrate_adaptive(f, (Q,), env; indicator=:absdelta, maxterms=16, atol=0.0, rtol=0.0)
        @test isapprox(val, 0.0 + (2 / 3) * im; atol=1e-12, rtol=1e-12)
    end

    @testset "Slow-growth families: growth, nestedness, and basic accuracy" begin
        WLp = WeightedLejaPoints(LegendreBasis(), LegendreMeasure(); candidate_count=33)
        WLq = WeightedLejaQuadrature(LegendreBasis(), LegendreMeasure(); candidate_count=33)
        PGq = PseudoGaussQuadrature(LegendreBasis(), LegendreMeasure(); candidate_count=33)
        CC  = ClenshawCurtisQuadrature()

        for r in 0:4
            @test qsize(WLp, r) == r + 1
            @test qsize(WLq, r) == r + 1
            @test qsize(PGq, r) == r + 1
            @test qsize(CC, r) == (1 << r) + 1
        end
        @test qsize(WLq, 4) < qsize(CC, 4)
        @test qsize(PGq, 4) < qsize(CC, 4)

        for r in 1:4
            @test qpoints(WLp, r)[1:end-1] == qpoints(WLp, r - 1)
            @test qpoints(qrule(WLq, r))[1:end-1] == qpoints(qrule(WLq, r - 1))
            @test qpoints(qrule(PGq, r))[1:end-1] == qpoints(qrule(PGq, r - 1))
        end

        # Weighted-Leja quadrature should integrate low-degree polynomials exactly.
        rule_wl = qrule(WLq, 3) # 4 points => exact on degree <= 3 in the chosen basis space
        for k in 0:3
            val = _rule_apply(rule_wl, x -> x^k)
            exact = isodd(k) ? 0.0 : 2 / (k + 1)
            @test isapprox(val, exact; atol=1e-10, rtol=1e-10)
        end

        # Pseudo-Gauss should improve quickly on a smooth polynomial observable.
        err2 = abs(_rule_apply(qrule(PGq, 1), x -> x^2) - 2 / 3)
        err4 = abs(_rule_apply(qrule(PGq, 3), x -> x^2) - 2 / 3)
        @test err4 < err2
        @test err4 < 0.1

        # Difference rules telescope for the slow-growth nested quadrature families too.
        f(x) = exp(x) + x^2
        for Q in (WLq, PGq)
            lhs = sum(_rule_apply(qdiffrule(Q, j), f) for j in 0:3)
            rhs = _rule_apply(qrule(Q, 3), f)
            @test isapprox(lhs, rhs; atol=1e-10, rtol=1e-10)
        end
    end

    @testset "Threaded tensor kernel matches serial" begin
        if nthreads() == 1
            @test true
        else
            Q = ClenshawCurtisQuadrature()
            rules = (qrule(Q, 6), qrule(Q, 6))
            f(x) = exp(x[1] - x[2] / 3) + 0.25im * (x[1]^2 + x[2])
            serial, npts_serial = UnifiedSparseGrids._tensor_rule_integral(f, rules; threaded=false)
            threaded, npts_threaded = UnifiedSparseGrids._tensor_rule_integral(f, rules; threaded=true)
            @test npts_serial == npts_threaded == 65^2
            @test isapprox(threaded, serial; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "Threaded nonnested mask reduction matches serial" begin
        if nthreads() == 1
            @test true
        else
            Q = GaussLegendreQuadrature()
            fams = (Q, Q, Q)
            f(x) = cos(x[1] - x[2]) + 0.1im * (x[1] * x[2] + x[3]^2)
            r = SVector(2, 1, 1)
            threaded, _, _ = delta_contribution(f, fams, r)
            manual = _manual_tensor_inclusion_exclusion(f, fams, r)
            @test isapprox(threaded, manual; atol=1e-12, rtol=1e-12)
        end
    end
end
