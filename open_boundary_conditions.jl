using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: Open

import Oceananigans.BoundaryConditions: getbc

struct TimeInterpolatedArray{I, A}
    interpolation_time :: I
    interpolated_array :: A
end

@inline function getbc(bc::BC{<:Open, <:TimeInterpolatedArray}, i::Integer, j::Integer, grid::AbstractGrid, clock::Clock, args...)
    time = clock.time

    n  = mod(time, bc.condition.interpolation_time) + 1
    n₁ = Int(floor(n))
    n₂ = Int(n₁ + 1)    
    
    return bc.condition.interpolated_array[i, j, n₁] * (n₂ - n) + bc.condition.interpolated_array[i, j, n₂] * (n - n₁)
end
    
function set_boundary_conditions()
    return NamedTuple()
end

function set_forcings()
    return NamedTuple()
end