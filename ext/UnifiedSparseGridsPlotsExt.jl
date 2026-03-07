module UnifiedSparseGridsPlotsExt

using UnifiedSparseGrids
import Plots

import UnifiedSparseGrids: plot_subspace_layout, plot_combination_technique,
                           plot_sparse_grid, plot_sparse_indexset

@inline function _square_limits(x::AbstractVector, y::AbstractVector)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    span = max(float(xmax - xmin), float(ymax - ymin))
    span == 0 && (span = 1.0)
    xmid = (float(xmax) + float(xmin)) / 2
    ymid = (float(ymax) + float(ymin)) / 2
    r = span / 2
    pad = 0.05 * span
    r += pad
    return (xmid - r, xmid + r), (ymid - r, ymid + r)
end

@inline function _square_limits(x::AbstractVector, y::AbstractVector, z::AbstractVector)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    zmin, zmax = extrema(z)
    span = maximum((float(xmax - xmin), float(ymax - ymin), float(zmax - zmin)))
    span == 0 && (span = 1.0)
    xmid = (float(xmax) + float(xmin)) / 2
    ymid = (float(ymax) + float(ymin)) / 2
    zmid = (float(zmax) + float(zmin)) / 2
    r = span / 2
    pad = 0.05 * span
    r += pad
    return (xmid - r, xmid + r), (ymid - r, ymid + r), (zmid - r, zmid + r)
end

@inline function _starts_by_level(nodes::AbstractUnivariateNodes, maxlev::Int)
    starts = Vector{Int}(undef, maxlev + 1)
    @inbounds for ℓ in 0:maxlev
        starts[ℓ + 1] = (ℓ == 0) ? 1 : (npoints(nodes, ℓ - 1) + 1)
    end
    return starts
end

function _plot_points(grid::SparseGrid;
                      space::Symbol,
                      square_axes::Bool,
                      xlabel::AbstractString,
                      ylabel::AbstractString,
                      zlabel::Union{Nothing,AbstractString}=nothing)
    D = dim(grid)
    (D == 2 || D == 3) || throw(ArgumentError("plot supports D==2 or D==3, got D=$D"))
    (space === :physical || space === :index) ||
        throw(ArgumentError("space must be :physical or :index, got $space"))

    nodes = grid.spec.axes
    caps = refinement_caps(grid)
    blocks = collect(each_subspace_block(grid))
    n = sum(b.len for b in blocks)
    n > 0 || throw(ArgumentError("grid is empty"))

    starts = ntuple(d -> _starts_by_level(nodes[d], caps[d]), Val(D))
    xfull = (space === :physical) ? ntuple(d -> points(nodes[d], caps[d]), Val(D)) : nothing

    group = Vector{Int}(undef, n)
    ms = Vector{Int}(undef, n)

    if D == 2
        if space === :physical
            x1 = Vector{Float64}(undef, n)
            x2 = Vector{Float64}(undef, n)
            i = 1
            @inbounds for b in blocks
                ℓ = b.refinement
                m = b.extents
                s = sum(ℓ)
                msval = max(1, 8 - s)
                s1 = starts[1][ℓ[1] + 1]
                s2 = starts[2][ℓ[2] + 1]
                for k2 in 0:(m[2] - 1), k1 in 0:(m[1] - 1)
                    x1[i] = xfull[1][s1 + k1]
                    x2[i] = xfull[2][s2 + k2]
                    group[i] = s
                    ms[i] = msval
                    i += 1
                end
            end
            xlims = ylims = nothing
            if square_axes
                xlims, ylims = _square_limits(x1, x2)
            end
            return Plots.scatter(x1, x2; group=group, markersize=ms, markerstrokewidth=0,
                                 legend=false, grid=false, aspect_ratio=:equal,
                                 xlims=xlims, ylims=ylims,
                                 xlabel=xlabel, ylabel=ylabel)
        else
            j1 = Vector{Int}(undef, n)
            j2 = Vector{Int}(undef, n)
            i = 1
            @inbounds for b in blocks
                ℓ = b.refinement
                m = b.extents
                s = sum(ℓ)
                msval = max(1, 8 - s)
                s1 = starts[1][ℓ[1] + 1]
                s2 = starts[2][ℓ[2] + 1]
                for k2 in 0:(m[2] - 1), k1 in 0:(m[1] - 1)
                    j1[i] = (s1 + k1) - 1
                    j2[i] = (s2 + k2) - 1
                    group[i] = s
                    ms[i] = msval
                    i += 1
                end
            end
            xlims = ylims = nothing
            if square_axes
                xlims, ylims = _square_limits(j1, j2)
            end
            return Plots.scatter(j1, j2; group=group, markersize=ms, markerstrokewidth=0,
                                 legend=false, grid=false, aspect_ratio=:equal,
                                 xlims=xlims, ylims=ylims,
                                 xlabel=xlabel, ylabel=ylabel)
        end
    else
        if space === :physical
            x1 = Vector{Float64}(undef, n)
            x2 = Vector{Float64}(undef, n)
            x3 = Vector{Float64}(undef, n)
            i = 1
            @inbounds for b in blocks
                ℓ = b.refinement
                m = b.extents
                s = sum(ℓ)
                msval = max(1, 8 - s)
                s1 = starts[1][ℓ[1] + 1]
                s2 = starts[2][ℓ[2] + 1]
                s3 = starts[3][ℓ[3] + 1]
                for k3 in 0:(m[3] - 1), k2 in 0:(m[2] - 1), k1 in 0:(m[1] - 1)
                    x1[i] = xfull[1][s1 + k1]
                    x2[i] = xfull[2][s2 + k2]
                    x3[i] = xfull[3][s3 + k3]
                    group[i] = s
                    ms[i] = msval
                    i += 1
                end
            end
            xlims = ylims = zlims = nothing
            if square_axes
                xlims, ylims, zlims = _square_limits(x1, x2, x3)
            end
            return Plots.scatter(x1, x2, x3; group=group, markersize=ms, markerstrokewidth=0,
                                 legend=false, grid=false,
                                 xlims=xlims, ylims=ylims, zlims=zlims,
                                 xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
        else
            j1 = Vector{Int}(undef, n)
            j2 = Vector{Int}(undef, n)
            j3 = Vector{Int}(undef, n)
            i = 1
            @inbounds for b in blocks
                ℓ = b.refinement
                m = b.extents
                s = sum(ℓ)
                msval = max(1, 8 - s)
                s1 = starts[1][ℓ[1] + 1]
                s2 = starts[2][ℓ[2] + 1]
                s3 = starts[3][ℓ[3] + 1]
                for k3 in 0:(m[3] - 1), k2 in 0:(m[2] - 1), k1 in 0:(m[1] - 1)
                    j1[i] = (s1 + k1) - 1
                    j2[i] = (s2 + k2) - 1
                    j3[i] = (s3 + k3) - 1
                    group[i] = s
                    ms[i] = msval
                    i += 1
                end
            end
            xlims = ylims = zlims = nothing
            if square_axes
                xlims, ylims, zlims = _square_limits(j1, j2, j3)
            end
            return Plots.scatter(j1, j2, j3; group=group, markersize=ms, markerstrokewidth=0,
                                 legend=false, grid=false,
                                 xlims=xlims, ylims=ylims, zlims=zlims,
                                 xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
        end
    end
end

function plot_subspace_layout(grid::SparseGrid;
                              annotate_offsets::Bool=true,
                              show_points::Bool=true,
                              show_diagonals::Bool=false,
                              show_ghost_blocks::Bool=false,
                              mark_points=nothing)
    D = dim(grid)
    D == 2 || throw(ArgumentError("plot_subspace_layout currently supports D==2, got D=$D"))

    p = Plots.plot(; aspect_ratio=:equal, legend=false, grid=false,
                   axis=false, framestyle=:none)

    blocks = collect(each_subspace_block(grid))
    blockmap = Dict{Tuple{Int,Int},SubspaceBlock{2}}(Tuple(b.refinement) => b for b in blocks)

    ghost_blocks = SubspaceBlock{2}[]
    if show_ghost_blocks
        I = grid.spec.indexset
        if I isa SmolyakIndexSet{2}
            I0 = SmolyakIndexSet(2, I.L)
            spec0 = SparseGridSpec(grid.spec.axes, I0)
            for b in each_subspace_block(SparseGrid(spec0))
                (b.refinement in I) && continue
                push!(ghost_blocks, b)
            end
        elseif I isa WeightedSmolyakIndexSet{2}
            I0 = WeightedSmolyakIndexSet(2, I.L, I.weights)
            spec0 = SparseGridSpec(grid.spec.axes, I0)
            for b in each_subspace_block(SparseGrid(spec0))
                (b.refinement in I) && continue
                push!(ghost_blocks, b)
            end
        end
    end

    all_blocks = show_ghost_blocks ? vcat(blocks, ghost_blocks) : blocks
    origin = if isempty(all_blocks)
        (0, 0)
    else
        o1 = minimum(b -> b.refinement[1], all_blocks)
        o2 = minimum(b -> b.refinement[2], all_blocks)
        (o1, o2)
    end

    function draw_block!(b::SubspaceBlock{2}; linealpha::Real, show_points::Bool, annotate::Bool)
        ℓ = b.refinement

        x0 = ℓ[2] - origin[2]
        y0 = ℓ[1] - origin[1]

        xs = [x0, x0 + 1, x0 + 1, x0, x0]
        ys = [y0, y0, y0 + 1, y0 + 1, y0]
        Plots.plot!(p, xs, ys; seriestype=:shape, fillalpha=0.0,
                    linealpha=linealpha, linecolor=:black)

        if annotate
            off = b.offset - 1
            Plots.annotate!(p, x0 + 0.06, y0 + 0.94, Plots.text(string(off), 8))
        end

        if show_points
            m1, m2 = b.extents
            n = m1 * m2
            if n > 0
                px = Vector{Float64}(undef, n)
                py = Vector{Float64}(undef, n)
                i = 1
                @inbounds for k2 in 0:(m2 - 1), k1 in 0:(m1 - 1)
                    px[i] = x0 + (k1 + 0.5) / m1
                    py[i] = y0 + (k2 + 0.5) / m2
                    i += 1
                end
                Plots.scatter!(p, px, py; markersize=3, markerstrokewidth=0,
                               color=:black)
            end
        end
    end

    for b in ghost_blocks
        draw_block!(b; linealpha=0.2, show_points=false, annotate=false)
    end
    for b in blocks
        draw_block!(b; linealpha=1.0, show_points=show_points, annotate=annotate_offsets)
    end

    if show_diagonals
        I = grid.spec.indexset
        if I isa SmolyakIndexSet{2}
            L = Int(I.L) - (origin[1] + origin[2])
            if L >= 0
                for s in 0:L
                    Plots.plot!(p, [0, s], [s, 0]; linestyle=:dash,
                                linealpha=0.6, linecolor=:black)
                end
            end
        end
    end

    if mark_points !== nothing
        nodes = grid.spec.axes
        caps = refinement_caps(grid)
        starts = (_starts_by_level(nodes[1], caps[1]), _starts_by_level(nodes[2], caps[2]))

        function mark_by_refinement_local!(r::NTuple{2,Int}, k::NTuple{2,Int})
            b = blockmap[r]

            x0 = b.refinement[2] - origin[2]
            y0 = b.refinement[1] - origin[1]
            m1, m2 = b.extents
            x = x0 + (k[1] + 0.5) / m1
            y = y0 + (k[2] + 0.5) / m2
            Plots.scatter!(p, [x], [y]; markersize=6, markerstrokewidth=0)
        end

        function mark_by_subspace_index!(idx::Int)
            idx >= 1 || return
            for b in blocks
                r = range(b)
                idx in r || continue
                lin = idx - first(r)
                m1, m2 = b.extents
                k1 = lin % m1
                k2 = lin ÷ m1
                return mark_by_refinement_local!(Tuple(b.refinement), (k1, k2))
            end
            return
        end

        function mark_by_coords!(c)
            r1 = searchsortedlast(starts[1], Int(c[1])) - 1
            r2 = searchsortedlast(starts[2], Int(c[2])) - 1
            (0 <= r1 <= caps[1] && 0 <= r2 <= caps[2]) || return

            prev1 = (r1 == 0) ? 0 : totalsize(nodes[1], r1 - 1)
            prev2 = (r2 == 0) ? 0 : totalsize(nodes[2], r2 - 1)
            k1 = Int(c[1]) - prev1 - 1
            k2 = Int(c[2]) - prev2 - 1

            r = (r1, r2)
            haskey(blockmap, r) || return
            return mark_by_refinement_local!(r, (k1, k2))
        end

        if mark_points isa AbstractVector{<:Integer}
            for idx in mark_points
                mark_by_subspace_index!(Int(idx))
            end
        elseif mark_points isa AbstractVector && !isempty(mark_points) && first(mark_points) isa AbstractVector
            for c in mark_points
                mark_by_coords!(c)
            end
        else
            for (ℓ_tup, k_tup) in mark_points
                ℓ = (Int(ℓ_tup[1]), Int(ℓ_tup[2]))
                k = (Int(k_tup[1]), Int(k_tup[2]))
                mark_by_refinement_local!(ℓ, k)
            end
        end
    end

    return p
end

function plot_combination_technique(d::Integer, n::Integer;
                                    cap=nothing,
                                    annotate_weights::Bool=true)
    d == 2 || throw(ArgumentError("plot_combination_technique currently supports d==2, got d=$d"))

    subs = each_combination_subproblem(d, n; cap=cap)
    p = Plots.plot(; aspect_ratio=:equal, legend=false, grid=false,
                   xlabel="ℓ₂", ylabel="ℓ₁", framestyle=:box)

    for sp in subs
        ℓ = sp.cap
        x1 = ℓ[2] + 1
        y1 = ℓ[1] + 1
        xs = [0, x1, x1, 0, 0]
        ys = [0, 0, y1, y1, 0]
        Plots.plot!(p, xs, ys; seriestype=:shape, fillalpha=0.0, linealpha=1.0)
        if annotate_weights
            Plots.annotate!(p, x1 + 0.02, y1 - 0.5,
                            Plots.text(string(sp.weight), 10))
        end
    end

    return p
end

function plot_sparse_grid(grid::SparseGrid;
                          layout::AbstractLayout=RecursiveLayout(),
                          square_axes::Bool=true)
    return _plot_points(grid;
                        space=:physical,
                        square_axes=square_axes,
                        xlabel="x₁",
                        ylabel="x₂",
                        zlabel=(dim(grid) == 3 ? "x₃" : nothing))
end

function plot_sparse_indexset(grid::SparseGrid;
                              layout::AbstractLayout=RecursiveLayout(),
                              square_axes::Bool=true)
    return _plot_points(grid;
                        space=:index,
                        square_axes=square_axes,
                        xlabel="j₁",
                        ylabel="j₂",
                        zlabel=(dim(grid) == 3 ? "j₃" : nothing))
end

end # module UnifiedSparseGridsPlotsExt
