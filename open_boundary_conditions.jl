using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: Open, Value, BC, DCBC
import Oceananigans.BoundaryConditions: getbc
import Base: getindex

struct TimeInterpolatedArray{T, N, I} 
    time_array :: AbstractArray{T, N}
    unit_time  :: I
end

@inline function getindex(t::TimeInterpolatedArray{T, N},  clock::Clock, idx...) where {T, N}
    @inbounds begin
        n = clock.time / t.unit_time +1
        n₁ = Int(floor(n))

        if n₁ == n
            return getindex(t.time_array, idx..., n₁)
        end
        n₂ = Int(n₁ + 1)

        return getindex(t.time_array, idx..., n₁) * (n₂ - n) + getindex(t.time_array, idx..., n₂) * (n - n₁)
    end
end

@inline getbc(bc::BC{<:Open,  <:TimeInterpolatedArray}, i::Integer, j::Integer, grid, clock::Clock, args...) = bc.condition[i, j, clock]
@inline getbc(bc::BC{<:Value, <:TimeInterpolatedArray}, i::Integer, j::Integer, grid, clock::Clock, args...) = bc.condition[i, j, clock]

function set_boundary_conditions(grid; Nt = 30)

    Nx, Ny, Nz = size(grid)
    arch = architecture(grid)

    u_west  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    u_east  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    u_south = arch_array(arch, zeros(Nx+1, Nz, Nt+1))
    u_north = arch_array(arch, zeros(Nx+1, Nz, Nt+1))

    v_west  = arch_array(arch, zeros(Ny+1, Nz, Nt+1))
    v_east  = arch_array(arch, zeros(Ny+1, Nz, Nt+1))
    v_south = arch_array(arch, zeros(Nx, Nz, Nt+1))
    v_north = arch_array(arch, zeros(Nx, Nz, Nt+1))

    T_west  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    T_east  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    T_south = arch_array(arch, zeros(Nx, Nz, Nt+1))
    T_north = arch_array(arch, zeros(Nx, Nz, Nt+1))

    S_west  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    S_east  = arch_array(arch, zeros(Ny, Nz, Nt+1))
    S_south = arch_array(arch, zeros(Nx, Nz, Nt+1))
    S_north = arch_array(arch, zeros(Nx, Nz, Nt+1))

    u_west_bc  =  OpenBoundaryCondition(TimeInterpolatedArray(u_west,  1day))
    u_east_bc  =  OpenBoundaryCondition(TimeInterpolatedArray(u_east,  1day))
    u_south_bc = ValueBoundaryCondition(TimeInterpolatedArray(u_south, 1day))
    u_north_bc = ValueBoundaryCondition(TimeInterpolatedArray(u_north, 1day))
    
    v_west_bc  = ValueBoundaryCondition(TimeInterpolatedArray(v_west,  1day))
    v_east_bc  = ValueBoundaryCondition(TimeInterpolatedArray(v_east,  1day))
    v_south_bc =  OpenBoundaryCondition(TimeInterpolatedArray(v_south, 1day))
    v_north_bc =  OpenBoundaryCondition(TimeInterpolatedArray(v_north, 1day))
    
    u_bcs = FieldBoundaryConditions(west = u_west_bc, east = u_east_bc, south = u_south_bc, north = u_north_bc)
    v_bcs = FieldBoundaryConditions(west = v_west_bc, east = v_east_bc, south = v_south_bc, north = v_north_bc)

    T_west_bc  = ValueBoundaryCondition(TimeInterpolatedArray(T_west,  1day))
    T_east_bc  = ValueBoundaryCondition(TimeInterpolatedArray(T_east,  1day))
    T_south_bc = ValueBoundaryCondition(TimeInterpolatedArray(T_north, 1day))
    T_north_bc = ValueBoundaryCondition(TimeInterpolatedArray(T_south, 1day))
    
    S_west_bc  = ValueBoundaryCondition(TimeInterpolatedArray(S_west,  1day))
    S_east_bc  = ValueBoundaryCondition(TimeInterpolatedArray(S_east,  1day))
    S_south_bc = ValueBoundaryCondition(TimeInterpolatedArray(S_north, 1day))
    S_north_bc = ValueBoundaryCondition(TimeInterpolatedArray(S_south, 1day))

    T_bcs = FieldBoundaryConditions(west = T_west_bc, east = T_east_bc, south = T_south_bc, north = T_north_bc)
    S_bcs = FieldBoundaryConditions(west = S_west_bc, east = S_east_bc, south = S_south_bc, north = S_north_bc)

    return (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs)
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
        west  = var.boundary_conditions.west.condition.time_array
        copyto!(west,  data.west)
    end

    if !(var.boundary_conditions.east isa DCBC)
        east  = var.boundary_conditions.east.condition.time_array
        copyto!(east,  data.east)
    end

    if !(var.boundary_conditions.south isa DCBC)
        south  = var.boundary_conditions.south.condition.time_array
        copyto!(south, data.south)
    end
    
    if !(var.boundary_conditions.north isa DCBC)
        north  = var.boundary_conditions.north.condition.time_array
        copyto!(north, data.north)
    end

    return nothing
end

function update_boundary_conditions!(simulation)

    u, v, w = simulation.model.velocities
    T, S, c = simulation.model.tracers

    # Decide what filename to load based on the clock (at the beginning just one file so open that!)

    @info "loading file $filename"

    file = jldopen(filename)

    # Partition the boundary data accordingly to arch!
    # Create the NamedTuple
    # Example:
    # data_u_west = extract_west(partition(read_from_binary(....)))
    # data_u_east = extract_east(partition(read_from_binary(....)))
    # bcs_u = (west = data_u_west, east = data_u_east, ....)
    # fill_boundaries!(u, bcs_u)
    
    # Better way:
    # Save all boundary conditions in .jld2 
    # with file["u"].west, file["u"].east... and so forth
    # and load with 
    # fill_boundaries!(u, file["u"])

    GC.gc()

    return nothing
end

#= fill_forcing!(var, data) = copyto!(var.parameters.time_array, data)

function update_forcing!(simulation)

    @info "loading file $filename"

    file = jldopen(filename)

    fill_forcing!(model.forcing.u, file["u"])
    fill_forcing!(model.forcing.v, file["v"])
    fill_forcing!(model.forcing.T, file["T"])
    fill_forcing!(model.forcing.S, file["S"])

    GC.gc()

    return nothing
end
 =#