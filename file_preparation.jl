using Oceananigans
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!

struct XDim end
struct YDim end

function read_from_binary(filename::String; Nx::Int=1, Ny::Int=1, Nz::Int=1, Nt::Int=1)
    # Initialize the array with 4 dimensions
    arr = zeros(Float32, Nx * Ny * Nz * Nt)

    # Read from the file
    read!(filename, arr)
    arr = bswap.(arr) .|> Float64
    arr = reshape(arr, Nx, Ny, Nz, Nt)

    # Always reverse in the z-direction
    arr = reverse(arr, dims=3)

    # Find which dimensions have size 1
    dims_to_drop = findall(size(arr) .== 1)
    
    # If there are any singleton dimensions, drop them
    if !isempty(dims_to_drop)
        return dropdims(arr; dims=tuple(dims_to_drop...))
    else
        return arr
    end
end

# Grid of threads with indices going from 1:Nx-1, 1:Ny, 1:Nz
@kernel function _reallocate_uv!(new_field, field, ::XDim)
    i′, j, k = @index(Global, NTuple) # launched with i in 1:Nx-1
    i = i′ + 1 # i becomes 2:Nx
    @inbounds new_field[i, j, k] = 0.5 * (field[i-1, j, k] + field[i, j, k])
end


@kernel function _reallocate_uv!(new_field, field, ::YDim)
    i, j′, k = @index(Global, NTuple)
    j = j′ + 1 # j becomes 2:Ny
    @inbounds new_field[i, j, k] = 0.5 * (field[i, j-1, k] + field[i, j, k])
end

@kernel function _correct_boundary_x!(new_field, field, Nx)
    j, k = @index(Global, NTuple)
    @inbounds new_field[1,    j, k] = (field[1,  j, k] * 3 - field[2,    j, k]) / 2
    @inbounds new_field[Nx+1, j, k] = (field[Nx, j, k] * 3 - field[Nx-1, j, k]) / 2
    
end

@kernel function _correct_boundary_y!(new_field, field, Ny)
    i, k = @index(Global, NTuple)
    @inbounds new_field[i, 1,    k] = (field[i, 1, k ] * 3 - field[i, 2,    k]) / 2
    @inbounds new_field[i, Ny+1, k] = (field[i, Ny, k] * 3 - field[i, Ny-1, k]) / 2
  
end

function reallocate_uv(field::AbstractArray{T, 3}; dim::Int=1) where T
    Nx, Ny, Nz = size(field)
    arch = Oceananigans.architecture(field)

    new_field_size = (Nx, Ny, Nz)
    if dim == 1
        new_field_size = (Nx+1, Ny, Nz)
    elseif dim == 2
        new_field_size = (Nx, Ny+1, Nz)
    else
        throw(ArgumentError("Invalid dimension specified. Please choose between 1, 2."))
    end
    new_field = arch_array(arch, zeros(new_field_size))

    if dim == 1
        reallocate_uv! = _reallocate_uv!(device(arch), (16, 16), (Nx-1, Ny, Nz))
        reallocate_uv!(new_field, field, XDim())
        correct_boundary_x! = _correct_boundary_x!(device(arch), (16, 16), (Ny, Nz))
        correct_boundary_x!(new_field, field, Nx)
    elseif dim == 2
        reallocate_uv! = _reallocate_uv!(device(arch), (16, 16), (Nx, Ny-1, Nz))
        reallocate_uv!(new_field, field, YDim())
        correct_boundary_y! = _correct_boundary_y!(device(arch), (16, 16), (Nx, Nz))
        correct_boundary_y!(new_field, field, Ny)
    else
        throw(ArgumentError("Invalid dimension specified. Please choose between 1, 2."))
    end

    return new_field
end

function setup_boundary_data!(path, grid; frequency = 1day, Nt = 158)
    times = 0:frequency:Nt*frequency

    save_variable!(path, "u", "Uvel", times, (Nothing, Center, Center), (Face,   Nothing, Center), size(grid)...)
    save_variable!(path, "v", "VVel", times, (Nothing, Face,   Center), (Center, Nothing, Center), size(grid)...)
    save_variable!(path, "T", "Temp", times, (Nothing, Center, Center), (Center, Nothing, Center), size(grid)...)
    save_variable!(path, "S", "Salt", times, (Nothing, Center, Center), (Center, Nothing, Center), size(grid)...)

    return nothing    
end

function save_variable!(path, var, var_mitgcm, times, locx, locy, Nx, Ny, Nz; Nt = length(times))

    west  = FieldTimeSeries{locx...}(grid, times; backend = OnDisk(), path, name = var * "_west")
    east  = FieldTimeSeries{locx...}(grid, times; backend = OnDisk(), path, name = var * "_east")
    south = FieldTimeSeries{locy...}(grid, times; backend = OnDisk(), path, name = var * "_south")
    north = FieldTimeSeries{locy...}(grid, times; backend = OnDisk(), path, name = var * "_north")

    tmpW = Field{locx...}(grid)
    tmpE = Field{locx...}(grid)
    tmpS = Field{locy...}(grid)
    tmpN = Field{locy...}(grid)

    for t in 1:Nt
        data_west  = read_from_binary("RT_100th" * var_mitgcm * "_W"; Ny, Nz, Nt)
        data_east  = read_from_binary("RT_100th" * var_mitgcm * "_E"; Ny, Nz, Nt)

        data_west  = locx[2] == Face ? reallocate_uv(data_west, dim = 2) : data_west
        data_east  = locx[2] == Face ? reallocate_uv(data_east, dim = 2) : data_east

        data_south = read_from_binary("RT_100th" * var_mitgcm * "_S"; Nx, Nz, Nt)
        data_north = read_from_binary("RT_100th" * var_mitgcm * "_N"; Nx, Nz, Nt)

        data_south = locy[2] == Face ? reallocate_uv(data_south, dim = 1) : data_south
        data_north = locy[2] == Face ? reallocate_uv(data_north, dim = 1) : data_north

        set!(tmpW, data_west)
        set!(tmpE, data_east)
        set!(tmpS, data_south)
        set!(tmpN, data_north)

        set!(west,  tmpx, t)
        set!(east,  tmpx, t)
        set!(south, tmpy, t)
        set!(north, tmpy, t)
    end

    return nothing
end



