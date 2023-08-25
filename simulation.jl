using MPI

const using_MPI = true

if using_MPI
    MPI.Init()
end

using Oceananigans
using Oceananigans.Units
using Oceananigans.Distributed
using Oceananigans.Distributed: partition_global_array
using JLD2 
using Printf
using SeawaterPolynomials
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using Oceananigans.Grids: architecture

#Yuchen's first change to the code
#Yuchen's second change to the code


include("open_boundary_conditions.jl")
include("grid_load_balance.jl")

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



@inline partition_array(arch::DistributedArch, array, size) = partition_global_array(arch, array, size)
@inline partition_array(arch, array, size) = array

@info "Creating the GPU grid"

topo = (Bounded, Bounded, Bounded)

if using_MPI
    Nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    arch = DistributedArch(GPU(); ranks = (Nranks, 1, 1), topology = topo)
else
    Nranks = 1
    rank = 0
    arch = GPU()
end

# grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))

Nx = 700
Ny = 500
Nz = 350

grid = grid_load_balance(arch, Nx, Ny, Nz, topo)

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

kappa = partition_array(arch, read_from_binary("data/RT_kappa_100th",Nx,Ny,Nz), size(grid))

vertical_diffusivity  = VerticalScalarDiffusivity(ν = kappa, κ = kappa)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 0.1)
# convective_adjustment = RiBasedVerticalDiffusivity()
# convective_adjustment = CATKEVerticalDiffusivity() 

equation_of_state = SeawaterPolynomials.TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state)

#####
##### Boundary conditions and forcing
#####

chunk_size_boundary = 30
chunk_size_forcing  = 4

boundary_conditions = NamedTuple() # set_boundary_conditions(grid; Nt = chunk_size)
forcing             = NamedTuple() # set_forcing(grid; Nt = chunk_size_forcing)

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

@info "Setting initial conditions"

u, v, w = model.velocities
T, S, c = model.tracers

u_init = zeros(size(u))
v_init = zeros(size(v))

# u_init[1:end-1, :, :] .= 
# v_init[:, 1:end-1, :] .= partition_array(arch, file_init["v"], size(T))

# set!(u, u_init)
# set!(v, v_init)
set!(T, partition_array(arch, read_from_binary("data/RT_TempInit_100th",Nx,Ny,Nz), size(T)))
set!(S, partition_array(arch, read_from_binary("data/RT_SaltInit_100th",Nx,Ny,Nz), size(S)))
set!(c, partition_array(arch, read_from_binary("data/TracerIC_RT_100th.bin",Nx,Ny,Nz), size(c)))



#####
##### Simulation and Diagnostics
##### 

simulation = Simulation(model; Δt = fixed_Δt, stop_iteration=200)

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

update_boundary_conditions!(simulation)

simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(20))

simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = TimeInterval(30days),
                                                        prefix = "Output_highres/RT_tracer_checkpoint_$(rank)",
                                                        overwrite_existing = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, w, T, S, c);
                                                      schedule = TimeInterval(day),
                                                      filename = "Output_highres/RT_tracer_fields_$(rank)",
                                                      overwrite_existing = true,
                                                      with_halos = true)

# simulation.callbacks[:update_boundary] = Callback(update_boundary_conditions!, TimeInterval(chunk_size_boundary*days))
# simulation.callbacks[:update_forcing]  = Callback(update_forcing!, TimeInterval(chunk_size_forcing*5days))

@info "Ready to run!!!"

# Let's RUN!!!set
run!(simulation)


for time in 0:simulation.stop_time
    time_step!(simulation.model, simulation.Δt)

    if need_to_callback
        execute_callback!(simulation)
    end
end