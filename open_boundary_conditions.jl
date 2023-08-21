using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: Open, BC

import Oceananigans.BoundaryConditions: getbc
import Base: getindex

struct TimeInterpolatedArray{A, I, T} 
    interpolated_array :: A
    interpolation_time :: I
    total_time :: T
end

@inline function getindex(t::TimeInterpolatedArray, i::Integer, j::Integer, clock::Clock)
    @inbounds begin
        time = clock.time
        n  = mod(time, t.interpolation_time) + 1
        n₁ = Int(floor(n))
        n₂ = Int(n₁ + 1) 
    
        return t.interpolated_array[i, j, n₁] * (n₂ - n) + t.interpolated_array[i, j, n₂] * (n - n₁)
    end
end

@inline getbc(bc::BC{<:Open, <:TimeInterpolatedArray}, i::Integer, j::Integer, grid, clock::Clock, args...) = bc.condition[i, j, clock]

function set_boundary_conditions()

    # u_west  = OpenBoundaryCondition(TimeInterpolatedArray(u_west_bc,  1day))
    # u_east  = OpenBoundaryCondition(TimeInterpolatedArray(u_east_bc,  1day))
    # u_south = OpenBoundaryCondition(TimeInterpolatedArray(u_north_bc, 1day))
    # u_north = OpenBoundaryCondition(TimeInterpolatedArray(u_south_bc, 1day))
    
    # v_west  = OpenBoundaryCondition(TimeInterpolatedArray(v_west_bc,  1day))
    # v_east  = OpenBoundaryCondition(TimeInterpolatedArray(v_east_bc,  1day))
    # v_south = OpenBoundaryCondition(TimeInterpolatedArray(v_north_bc, 1day))
    # v_north = OpenBoundaryCondition(TimeInterpolatedArray(v_south_bc, 1day))
    


    return NamedTuple()
end

function set_forcing()
    return NamedTuple()
end
