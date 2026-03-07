# Generate static figures for the manual pages.
# Run with: julia --project=docs docs/generate_figures.jl

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using UnifiedSparseGrids

# Ensure a headless GR backend when running in CI / containers.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Plots

outdir = joinpath(@__DIR__, "src", "assets")
mkpath(outdir)

# Subspace layout / block offsets (2D only).
# Use interior dyadic nodes (no endpoints) to match the right-hand side block layout
# in Fig. 5 of the pseudorecursive layout paper (d=2, n=4).
# In this package, that corresponds to `SmolyakIndexSet(2, 6)` (L = 5) with
# interior levels starting at 1.
#
# Add an anisotropic cap to illustrate truncation.
grid_layout = SparseGrid(SparseGridSpec(ntuple(_ -> DyadicNodes(endpoints=false), Val(2)),
                                       SmolyakIndexSet(2, 6; cap=(5, 3))))

p1 = plot_subspace_layout(grid_layout;
                          annotate_offsets=true,
                          show_points=true,
                          show_diagonals=true,
                          show_ghost_blocks=true)
savefig(p1, joinpath(outdir, "subspace_layout.svg"))

# Sparse grid point set and its corresponding integer index set (2D).
# Use dyadic nodes *with* endpoints for a hat-basis style point set.
grid_pts = SparseGrid(SparseGridSpec(ntuple(_ -> DyadicNodes(), Val(2)),
                                    SmolyakIndexSet(2, 5)))
p_grid = plot_sparse_grid(grid_pts)
p_idx  = plot_sparse_indexset(grid_pts)
p2 = Plots.plot(p_grid, p_idx; layout=(1, 2))
savefig(p2, joinpath(outdir, "sparse_grid_indexset_2d.svg"))
