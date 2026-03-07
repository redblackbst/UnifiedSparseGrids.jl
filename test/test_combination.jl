
using Test
using StaticArrays
using UnifiedSparseGrids

import UnifiedSparseGrids: _enumerate_subspaces

function _net_weight(d::Int, subs, r)
    tot = 0
    @inbounds for sp in subs
        ok = true
        for j in 1:d
            if r[j] > sp.cap[j]
                ok = false
                break
            end
        end
        ok && (tot += sp.weight)
    end
    return tot
end

@testset "Weight pattern sanity" begin
    cases = (
        (2, 4, Dict(5 => 1, 4 => -1)),
        (3, 3, Dict(5 => 1, 4 => -2, 3 => 1)),
    )
    for (d, n, expected) in cases
        subs = each_combination_subproblem(d, n)
        sums = Set{Int}()
        for sp in subs
            s = sum(sp.cap)
            push!(sums, s)
            @test sp.weight == expected[s]
        end
        @test sort(collect(sums)) == sort(collect(keys(expected)))
    end
end

@testset "Net hierarchical weight sanity" begin
    for (d, n) in ((2, 4), (3, 3))
        subs = each_combination_subproblem(d, n)
        I = SmolyakIndexSet(d, n + d - 1)
        subspaces = _enumerate_subspaces(I, refinement_caps(I))
        for r in subspaces
            @test _net_weight(d, subs, r) == 1
        end
    end
end

@testset "Cap-respecting combination subproblems" begin
    d = 3
    n = 4
    cap = SVector(4, 2, 1)
    subs = each_combination_subproblem(d, n; cap=cap)
    for sp in subs
        @test all(sp.cap .<= cap)
    end
end
