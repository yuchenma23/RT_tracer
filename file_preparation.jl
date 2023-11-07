using Oceananigans
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!

struct XDim end
struct YDim end

function read_from_binary(filename::String; Nx::Int=1, Ny::Int=1, Nz::Int=1, Nt::Int=1)
    # Initialize the array with 4 dimensions
    arr = zeros(Float32, Nx, Ny, Nz, Nt)

    # Read from the file
    read!(filename, arr)
    arr = bswap.(arr) .|> Float64

    # Always reverse in the z-direction
    arr= reverse(arr, dims=3)

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
