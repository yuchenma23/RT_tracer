using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: Open, Value, BC, DCBC, AbstractBoundaryConditionClassification
import Oceananigans.BoundaryConditions: getbc
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device, CPU, GPU, array_type, arch_array
import Base: getindex

using Adapt

function set_boundary_conditions_and_restoring(; chunk_size = 20)

    u_fts = load_boundary_conditions("u"; chunk_size)
    v_fts = load_boundary_conditions("v"; chunk_size)
    T_fts = load_boundary_conditions("T"; chunk_size)
    S_fts = load_boundary_conditions("S"; chunk_size)

    u_west_bc  =  OpenBoundaryCondition(u_fts.west )
    u_east_bc  =  OpenBoundaryCondition(u_fts.east )
    u_south_bc = ValueBoundaryCondition(u_fts.south)
    u_north_bc = ValueBoundaryCondition(u_fts.north)
    
    v_west_bc  = ValueBoundaryCondition(v_fts.west )
    v_east_bc  = ValueBoundaryCondition(v_fts.east )
    v_south_bc =  OpenBoundaryCondition(v_fts.south)
    v_north_bc =  OpenBoundaryCondition(v_fts.north)
    
    u_bcs = FieldBoundaryConditions(west = u_west_bc, east = u_east_bc, south = u_south_bc, north = u_north_bc)
    v_bcs = FieldBoundaryConditions(west = v_west_bc, east = v_east_bc, south = v_south_bc, north = v_north_bc)

    T_west_bc  = ValueBoundaryCondition(T_fts.west )
    T_east_bc  = ValueBoundaryCondition(T_fts.east )
    T_south_bc = ValueBoundaryCondition(T_fts.north)
    T_north_bc = ValueBoundaryCondition(T_fts.south)
    
    S_west_bc  = ValueBoundaryCondition(S_fts.west )
    S_east_bc  = ValueBoundaryCondition(S_fts.east )
    S_south_bc = ValueBoundaryCondition(S_fts.north)
    S_north_bc = ValueBoundaryCondition(S_fts.south)

    T_bcs = FieldBoundaryConditions(west = T_west_bc, east = T_east_bc, south = T_south_bc, north = T_north_bc)
    S_bcs = FieldBoundaryConditions(west = S_west_bc, east = S_east_bc, south = S_south_bc, north = S_north_bc)

    return (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs)
end

function load_boundary_conditions(var; chunk_size)
    west  = FieldTimeSeries("boundary_conditions.jld2", var * "_west";  backend = InMemory(; chunk_size))
    east  = FieldTimeSeries("boundary_conditions.jld2", var * "_east";  backend = InMemory(; chunk_size))
    south = FieldTimeSeries("boundary_conditions.jld2", var * "_south"; backend = InMemory(; chunk_size))
    north = FieldTimeSeries("boundary_conditions.jld2", var * "_north"; backend = InMemory(; chunk_size))

    return (; west, east, south, north)
end

#= @inline function relaxation_forcing(i, j, k, grid, clock, fields, p)
    C★ = p.forcing[i, j, k, clock]
    C  = getproperty(fields, p.variable)[i, j, k]
    return 1 / p.λ * p.mask[i, j, k] * (C★ - C)
end

function set_forcing(grid; Nt = 2)
    mask = arch_array(arch, zeros(Nx, Ny, Nz))

    mask .= jldopen("mask.jld2")["mask"]

    u_array = TimeInterpolatedArray(arch_array(arch, zeros(Nx, Ny, Nz, Nt+1)), 5days)
    v_array = TimeInterpolatedArray(arch_array(arch, zeros(Nx, Ny, Nz, Nt+1)), 5days)
    T_array = TimeInterpolatedArray(arch_array(arch, zeros(Nx, Ny, Nz, Nt+1)), 5days)
    S_array = TimeInterpolatedArray(arch_array(arch, zeros(Nx, Ny, Nz, Nt+1)), 5days)

    u_forcing = Forcing(relaxation_forcing, discrete_form=true, parameters = (mask = mask, variable = :u, λ = 5days, forcing = u_array))
    v_forcing = Forcing(relaxation_forcing, discrete_form=true, parameters = (mask = mask, variable = :v, λ = 5days, forcing = v_array))
    T_forcing = Forcing(relaxation_forcing, discrete_form=true, parameters = (mask = mask, variable = :T, λ = 5days, forcing = T_array))
    S_forcing = Forcing(relaxation_forcing, discrete_form=true, parameters = (mask = mask, variable = :S, λ = 5days, forcing = S_array))

    return (u = u_forcing, v = v_forcing, T = T_forcing, S = S_forcing)
end =#

function fill_boundaries!(var, data)

    # update BC only if not communicating
    if !(var.boundary_conditions.west isa DCBC)
        copyto!(var.boundary_conditions.west.condition.time_array,  data.west)
    end

    if !(var.boundary_conditions.east isa DCBC)
        copyto!(var.boundary_conditions.east.condition.time_array,  data.east)
    end

    if !(var.boundary_conditions.south isa DCBC)
        copyto!(var.boundary_conditions.south.condition.time_array, data.south)
    end
    
    if !(var.boundary_conditions.north isa DCBC)
        copyto!(var.boundary_conditions.north.condition.time_array, data.north)
    end

    return nothing
end

function setup_forcings_and_boundary_conditions!(grid; times = 1:1000)

    # Decide what filename to load based on the clock (at the beginning just one file so open that!)

    @info "loading binary files"

    #file = jldopen(filename)

    Nx, Ny, Nz = size(grid)

     
    data_u_east  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_u_west  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_u_north = arch_array(arch, zeros(Nx, Nz, Nt+1))
    data_u_south = arch_array(arch, zeros(Nx, Nz, Nt+1))

    data_v_east  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_v_west  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_v_north = arch_array(arch, zeros(Nx, Nz, Nt+1))
    data_v_south = arch_array(arch, zeros(Nx, Nz, Nt+1))

    data_T_east  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_T_west  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_T_north = arch_array(arch, zeros(Nx, Nz, Nt+1))
    data_T_south = arch_array(arch, zeros(Nx, Nz, Nt+1))

    data_S_east  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_S_west  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    data_S_north = arch_array(arch, zeros(Nx, Nz, Nt+1))
    data_S_south = arch_array(arch, zeros(Nx, Nz, Nt+1))

    data_u_east  = read_from_binary("data/RT_50thUvel_E"; Ny, Nz, Nt=Nt+1)
    data_u_west  = read_from_binary("data/RT_50thUvel_W"; Ny, Nz, Nt=Nt+1)
    data_u_north = partition_array(arch, read_from_binary("data/RT_50thUvel_N"; Nx=Nx, Nz=Nz, Nt=Nt+1), (Nx, Nz, Nt+1))
    data_u_south = partition_array(arch, read_from_binary("data/RT_50thUvel_S"; Nx=Nx, Nz=Nz, Nt=Nt+1), (Nx, Nz, Nt+1))

    
    data_v_east  = read_from_binary("data/RT_50thVvel_E"; Ny, Nz, Nt=Nt+1)
    data_v_west  = read_from_binary("data/RT_50thVvel_W"; Ny, Nz, Nt=Nt+1)
    data_v_north = partition_array(arch, read_from_binary("data/RT_50thVvel_N"; Nx, Nz, Nt=Nt+1), (Nx, Nz, Nt+1))
    data_v_south = partition_array(arch, read_from_binary("data/RT_50thVvel_S"; Nx, Nz, Nt=Nt+1), (Nx, Nz, Nt+1))


    data_S_east  = read_from_binary("data/RT_50thTemp_E"; Ny, Nz, Nt=Nt+1)
    data_S_west  = read_from_binary("data/RT_50thTemp_W"; Ny, Nz, Nt=Nt+1)
    data_S_north = partition_array(arch, read_from_binary("data/RT_50thTemp_N"; Nx, Nz, Nt=Nt+1), (Nx, Nz, Nt+1))
    data_S_south = partition_array(arch, read_from_binary("data/RT_50thTemp_S"; Nx, Nz, Nt=Nt+1), (Nx, Nz, Nt+1))


    data_T_east  = read_from_binary("data/RT_50thSalt_E"; Ny, Nz, Nt = Nt+1)
    data_T_west  = read_from_binary("data/RT_50thSalt_W"; Ny, Nz, Nt = Nt+1)
    data_T_north = partition_array(arch, read_from_binary("data/RT_50thSalt_N"; Nx, Nz, Nt=Nt+1), (Nx, Nz, Nt+1))
    data_T_south = partition_array(arch, read_from_binary("data/RT_50thSalt_S"; Nx, Nz, Nt=Nt+1), (Nx, Nz, Nt+1))

    # data_u_west = extract_west(partition(read_from_binary(....)))
    # data_u_east = extract_east(partition(read_from_binary(....)))

    #the function reallocate_uv extrapolated the center grid to the face grid


    bcs_u = (west = data_u_west,                       east = data_u_east,                         north=reallocate_uv(data_u_north;dim=1),    south=reallocate_uv(data_u_south;dim=1))
    bcs_v = (west = reallocate_uv(data_v_west;dim=2),  east = reallocate_uv(data_v_east;dim=2),    north=data_v_north,                         south=data_v_south)
    bcs_T = (west = data_T_west,                       east = data_T_east,                         north=data_T_north,                         south=data_T_south)
    bcs_S = (west = data_S_west,                       east = data_S_east,                         north=data_S_north,                         south=data_S_south)
     
    fill_boundaries!(u, bcs_u)
    fill_boundaries!(v, bcs_v)
    fill_boundaries!(T, bcs_T)
    fill_boundaries!(S, bcs_S)
    
    # Better way:
    # Save all boundary conditions in .jld2 
    # with file["u"].west, file["u"].east... and so forth
    # and load with 
    # fill_boundaries!(u, file["u"])

    GC.gc()

    return nothing
end