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


include("file_preparation.jl")
include("open_boundary_conditions.jl")


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

Nx_tot = 350
Nx = Nx_tot ÷ Nranks
Ny = 250
Nz = 175

grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz),
                               latitude = (53, 58), 
                              longitude = (-16.3, -9.3), 
                                      z = (-3500, 0),
                                   halo = (7, 7, 7),
                               topology = topo)

bottom = partition_array(arch, read_from_binary("data/RT_bathy_50th"; Nx=Nx_tot,Ny=Ny), size(grid))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom), true)
# grid = ImmersedBoundaryGrid(grid, PartialCellBottom(bottom))

#####
##### Numerics 
#####

@info "Defining numerics"

fixed_Δt = 100

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




kappa = partition_array(arch, read_from_binary("data/RT_kappa_50th"; Nx=Nx_tot,Ny=Ny,Nz=Nz), size(grid))
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

boundary_conditions = NamedTuple() # 
set_boundary_conditions(grid; Nt = 30)

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



T_init = read_from_binary("data/RT_TempInit_50th"; Nx=Nx_tot,Ny=Ny,Nz=Nz)
S_init = read_from_binary("data/RT_SaltInit_50th"; Nx=Nx_tot,Ny=Ny,Nz=Nz)
c_init = read_from_binary("data/TracerIC_RT_50th.bin"; Nx=Nx_tot,Ny=Ny,Nz=Nz)
u_init = read_from_binary("data/RT_UvelInit_50th"; Nx=Nx_tot,Ny=Ny,Nz=Nz)
v_init = read_from_binary("data/RT_VvelInit_50th"; Nx=Nx_tot,Ny=Ny,Nz=Nz)


# u_init[1:end-1, :, :] .= 
# v_init[:, 1:end-1, :] .= partition_array(arch, file_init["v"], size(T))

# set!(u, u_init)
# set!(v, v_init)



set!(u, partition_array(arch,reallocate_uv(u_init;dim=1) , size(u)))
set!(v, partition_array(arch,reallocate_uv(v_init;dim=2) , size(v)))
set!(T, partition_array(arch,T_init , size(T)))
set!(S, partition_array(arch,S_init , size(S)))
set!(c, partition_array(arch,c_init , size(c)))



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


update_boundary_conditions!(simulation)

simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(20))

simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = TimeInterval(30days),
                                                        prefix = "Output_lowres/RT_tracer_checkpoint_$(rank)",
                                                        overwrite_existing = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, w, T, S, c);
                                                      schedule = TimeInterval(day),
                                                      filename = "Output_lowres/RT_tracer_fields_$(rank)",
                                                      overwrite_existing = true,
                                                      with_halos = true)

# simulation.callbacks[:update_boundary] = Callback(update_boundary_conditions!, TimeInterval(chunk_size_boundary*days))
# simulation.callbacks[:update_forcing]  = Callback(update_forcing!, TimeInterval(chunk_size_forcing*5days))

@info "Ready to run!!!"

# Let's RUN!!!set
run!(simulation)
