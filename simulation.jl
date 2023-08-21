using Oceananigans
using Oceananigans.Units
using JLD2 
using Printf
using SeawaterPolynomials
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

function read_from_binary(filename, Nx, Ny, Nz)
    arr = zeros(Float32, Nx*Ny*Nz)
    read!(filename, arr)
    arr = bswap.(arr) .|> Float64
    arr = reshape(arr, Nx, Ny, Nz)

    return reverse(arr, dims = 3)
end


#####
##### Specifying domain, grid and bathymetry
#####

arch = GPU()
Nx = 700
Ny = 500
Nz = 350

grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz),
                               latitude = (53, 58), 
                              longitude = (-16.3, -9.3), 
                                      z = (-3500, 0),
                                   halo = (6, 6, 6),
                               topology = (Bounded, Bounded, Bounded))

bottom = jldopen("RT_bathy_100.jld2")["bathymetry"]

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map = true)
# grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))

#####
##### Numerics 
#####

fixed_Δt = 50

momentum_advection = VectorInvariant(vorticity_scheme = WENO(; order = 9), 
                                      vertical_scheme = WENO(),
                                    divergence_scheme = WENO())

tracer_advection = WENO(; order = 7)

free_surface = SplitExplicitFreeSurface(; cfl = 0.75, fixed_Δt, grid)
# free_surface = ImplicitFreeSurface()

coriolis = HydrostaticSphericalCoriolis()

#####
##### Physics
#####

kappa = jldopen("RT_kappa_100th.jld2")["kappa"]

vertical_diffusivity  = VerticalScalarDiffusivity(ν = kappa, κ = kappa)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 0.1)
# convective_adjustment = RiBasedVerticalDiffusivity()
# convective_adjustment = CATKEVerticalDiffusivity() 

equation_of_state = SeawaterPolynomials.TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state)

#####
##### Boundary conditions and forcing
#####



#####
##### Model construction and initialization
#####

model = HydrostaticFreeSurfaceModel(; grid, 
                                      momentum_advection,
                                      tracer_advection,
                                      coriolis,
                                      buoyancy,
                                      tracers = (:T, :S, :c),
                                      free_surface)

file_init = jldopen("initial_conditions.jld2")

u, v, w = model.velocities
T, S, c = model.tracers

set!(u, file_init["u"])
set!(v, file_init["v"])
set!(T, file_init["T"])
set!(S, file_init["S"])
set!(c, file_init["c"])

#####
##### Simulation and Diagnostics
##### 

simulation = Simulation(model; Δt = fixed_Δt, stop_time = 365days)

wall_time = Ref(time_ns())

function print_progress(simulation)
    model = simulation
    u, v, w = model.velocities
    T, S, c = model.tracers

    exec_time = (time_ns() - wall_time[]) * 1e9

    @sprintf("Iteration %d, wall_time %.2f, max(|u|, |v|, |w|): %.2f, %.2f, %.2f, max(T, S, c): %2.f, %.2f, %2.f\n", 
            model.clock.iteration, exec_time, maximum(abs, u), maximum(abs, v), maximum(abs, w), 
            maximum(T), maximum(S), maximum(c))
    
    wall_time[] = time_ns()
    
    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(20))

simulation.output_writer[:checkpointer] = Checkpointer(model; schedule = TimeInterval(30days),
                                                       prefix = "RT_tracer_checkpoint",
                                                       overwrite_existing = true)

simulation.output_writer[:fields] = JLD2OutputWriter(model, (; u, v, w, T, S, c);
                                                     schedule = TimeInterval(day),
                                                     filename = "RT_tracer_fields",
                                                     overwrite_existing = true,
                                                     with_halos = true)

# Let's RUN!!!
run!(simulation)