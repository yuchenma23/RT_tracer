using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: Open, BC

import Oceananigans.BoundaryConditions: getbc
import Base: getindex

struct TimeInterpolatedArray{A, I} 
    time_array :: A
    unit_time  :: I
end

@inline function getindex(t::TimeInterpolatedArray, i::Integer, j::Integer, clock::Clock)
    @inbounds begin
        time = clock.time / t.unit_time
        n    = mod(time, size(ta.time_array, 3)) + 1
        n₁   = Int(floor(n))

        if n₁ == n
            return t.time_array[i, j, n₁] 
        end

        n₂ = Int(n₁ + 1)
        return t.time_array[i, j, n₁] * (n₂ - n) + t.time_array[i, j, n₂] * (n - n₁)
    end
end

@inline getbc(bc::BC{<:Open, <:TimeInterpolatedArray}, i::Integer, j::Integer, grid, clock::Clock, args...) = bc.condition[i, j, clock]

function set_boundary_conditions(grid)

    Nx, Ny, Nz = size(grid)

    # u_west = arch_array()
    # u_west = arch_array()
    # u_west = arch_array()
    # u_west = arch_array()

    # u_west = arch_array()
    # u_west = arch_array()
    # u_west = arch_array()
    # u_west = arch_array()

    # u_west = arch_array()
    # u_west = arch_array()
    # u_ = arch_array()
    # u_ = arch_array()

    u_west_bc  =  OpenBoundaryCondition(TimeInterpolatedArray(u_west,  1day))
    u_east_bc  =  OpenBoundaryCondition(TimeInterpolatedArray(u_east,  1day))
    u_south_bc = ValueBoundaryCondition(TimeInterpolatedArray(u_south, 1day))
    u_north_bc = ValueBoundaryCondition(TimeInterpolatedArray(u_north, 1day))
    
    v_west_bc  = ValueBoundaryCondition(TimeInterpolatedArray(v_west,  1day))
    v_east_bc  = ValueBoundaryCondition(TimeInterpolatedArray(v_east,  1day))
    v_south_bc =  OpenBoundaryCondition(TimeInterpolatedArray(v_south, 1day))
    v_north_bc =  OpenBoundaryCondition(TimeInterpolatedArray(v_south, 1day))
    
    u_bcs = FieldBoundaryConditions(west = u_west_bc, east = u_east_bc, south = u_south_bc, north = u_north_bc)
    v_bcs = FieldBoundaryConditions(west = v_west_bc, east = v_east_bc, south = v_south_bc, north = v_north_bc)

    T_west_bc  = ValueBoundaryCondition(TimeInterpolatedArray(T_west,  1day))
    T_east_bc  = ValueBoundaryCondition(TimeInterpolatedArray(T_east,  1day))
    T_south_bc = ValueBoundaryCondition(TimeInterpolatedArray(T_north, 1day))
    T_north_bc = ValueBoundaryCondition(TimeInterpolatedArray(T_south, 1day))
    
    S_west_bc  = ValueBoundaryCondition(TimeInterpolatedArray(T_west,  1day))
    S_east_bc  = ValueBoundaryCondition(TimeInterpolatedArray(T_east,  1day))
    S_south_bc = ValueBoundaryCondition(TimeInterpolatedArray(T_north, 1day))
    S_north_bc = ValueBoundaryCondition(TimeInterpolatedArray(T_south, 1day))

    T_bcs = FieldBoundaryConditions(west = T_west_bc, east = T_east_bc, south = T_south_bc, north = T_north_bc)
    S_bcs = FieldBoundaryConditions(west = S_west_bc, east = S_east_bc, south = S_south_bc, north = S_north_bc)

    return NamedTuple() 
end

function set_forcing()
    return NamedTuple()
end