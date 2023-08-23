using KernelAbstractions: @kernel, @index
using Oceananigans: device
using Oceananigans.Grids: immersed_cell



function read_from_binary_2d(filename, Nx, Ny)
    arr = zeros(Float32, Nx*Ny)
    read!(filename, arr)
    arr = bswap.(arr) .|> Float64
    return reshape(arr, Nx, Ny)

end


function grid_load_balance(arch, Nx, Ny, Nz, topo)
    
    grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz),
                                latitude = (53, 58), 
                                longitude = (-16.3, -9.3), 
                                        z = (-3500, 0),
                                    halo = (7, 7, 7),
                                topology = topo)

    bottom = partition_array(arch, read_from_binary_2d("data/RT_bathy_100th",Nx,Ny), size(grid))

    return ImmersedBoundaryGrid(grid, GridFittedBottom(bottom), true)
end

function grid_load_balance(arch::DistributedArch, Nx, Ny, Nz, topo)

    grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz),
                                latitude = (53, 58), 
                                longitude = (-16.3, -9.3), 
                                        z = (-3500, 0),
                                    halo = (7, 7, 7),
                                topology = topo)

    bottom = partition_array(arch, read_from_binary_2d("data/RT_bathy_100th",Nx,Ny), size(grid))

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom), true)

    load_per_slab = arch_array(arch, zeros(Int, Nx))

    loop! = assess_load!(Oceananigans.Architectures.device(GPU()), 512, Nx)
    loop!(load_per_slab, grid)

    local_N = calculate_local_N(load_per_slab, N, arch)
    rank    = MPI.Comm_rank(MPI.COMM_WORLD)
    
    N = (local_N[rank+1], Ny, Nz)

    @info "slab decomposition on" rank N

    grid = LatitudeLongitudeGrid(arch; size = N,
                                latitude = (53, 58), 
                               longitude = (-16.3, -9.3), 
                                       z = (-3500, 0),
                                    halo = (7, 7, 7),
                                topology = topo)

    bottom = partition_array(arch, read_from_binary_2d("data/RT_bathy_100th",Nx,Ny), size(grid))

    return ImmersedBoundaryGrid(grid, GridFittedBottom(bottom), true)
end

@kernel function assess_load!(load_per_slab, grid)
    i = @index(Global, Linear)
    @unroll for j in 1:size(grid, 2)
        @unroll for k in 1:size(grid, 3)
            @inbounds load_per_slab[i] += ifelse(immersed_cell(i, j, k, grid), 0, 1)
        end
    end
end

function calculate_local_N(load_per_slab, N, arch)
    active_cells = sum(load_per_slab)
    active_load  = active_cells / arch.ranks[1]
    local_N      = zeros(Int, arch.ranks[1])
    idx = 1

    for r in 1:arch.ranks[1] - 1
        local_load = 0
        while local_load <= active_load
            local_load += load_per_slab[idx]
            local_N[r] += 1
            idx += 1
        end
    end

    local_N[end] = Nx - sum(local_N[1:end-1])

    return local_N
end