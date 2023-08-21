using Oceananigans
using Oceananigans.Units
using JLD2 
using Printf
using SeawaterPolynomials
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

include("open_boundary_conditions.jl")

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
#=
@info "Creating the GPU grid"

arch = GPU()
Nx = 700
Ny = 500
Nz = 350

grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz),
                               latitude = (53, 58), 
                              longitude = (-16.3, -9.3), 
                                      z = (-3500, 0),
                                   halo = (7, 7, 7),
                               topology = (Bounded, Bounded, Bounded))

bottom = jldopen("RT_bathy_100.jld2")["bathymetry"]

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom), true)
# grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))

#####
##### Numerics 
#####

@info "Defining numerics"

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

@info "Diffusivity and Buoyancy"

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

boundary_conditions = set_boundary_conditions()
forcing             = set_forcing()

#####
##### Model construction and initialization
#####

@info "Allocating the model"

model = HydrostaticFreeSurfaceModel(; grid, 
                                      momentum_advection,
                                      tracer_advection,
                                      coriolis,
                                      buoyancy,
                                      boundary_conditions,
                                      forcing,
                                      tracers = (:T, :S, :c),
                                      free_surface)
=#
@info "Setting initial conditions"

file_init = jldopen("initial_conditions.jld2")

u, v, w = model.velocities
T, S, c = model.tracers

u_init = zeros(size(u))
v_init = zeros(size(v))

u_init[1:end-1, :, :] .= file_init["u"]
v_init[:, 1:end-1, :] .= file_init["v"]

set!(u, u_init)
set!(v, v_init)
set!(T, file_init["T"])
set!(S, file_init["S"])
set!(c, file_init["c"])

#####
##### Simulation and Diagnostics
##### 

simulation = Simulation(model; Δt = fixed_Δt, stop_time = 365days)

wall_time = Ref(time_ns())

function print_progress(simulation)
    model = simulation.model
    u, v, w = model.velocities
    T, S, c = model.tracers

    exec_time = (time_ns() - wall_time[]) * 1e-9

    @info @sprintf("Iteration %d, wall_time %.2f, max(|u|, |v|, |w|): %.2f, %.2f, %.2f, max(T, S, c): %2.f, %.2f, %2.f\n", 
            model.clock.iteration, exec_time, maximum(abs, u), maximum(abs, v), maximum(abs, w), 
            maximum(T), maximum(S), maximum(c))
    
    wall_time[] = time_ns()
    
    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(20))

simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = TimeInterval(30days),
                                                        prefix = "RT_tracer_checkpoint",
                                                        overwrite_existing = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, w, T, S, c);
                                                      schedule = TimeInterval(day),
                                                      filename = "RT_tracer_fields",
                                                      overwrite_existing = true,
                                                      with_halos = true)

@info "Ready to run!!!"

# Let's RUN!!!set
run!(simulation)
