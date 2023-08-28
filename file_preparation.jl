
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


function reallocate_uv(field::AbstractArray{T, 3}; dim::Int=1) where T
    Nx, Ny, Nz = size(field)

    # Depending on the dimension specified by `dim`, we'll interpolate the data.
    # The interpolated field will have one additional element along the specified dimension.
    if dim == 1
        new_field = Array{T}(undef, Nx+1, Ny, Nz)
        for j=1:Ny, k=1:Nz
            new_field[1, j, k] = (field[1, j, k]*3-field[2, j, k])/2  #simple linear interpolation 
            new_field[Nx+1, j, k] = (field[Nx, j, k]*3-field[Nx-1, j, k])/2
            for i=2:Nx
                new_field[i, j, k] = 0.5 * (field[i-1, j, k] + field[i, j, k])
            end
        end
    elseif dim == 2
        new_field = Array{T}(undef, Nx, Ny+1, Nz)
        for i=1:Nx, k=1:Nz
            new_field[i, 1, k] = (field[i, 1, k]*3-field[i, 2, k])/2 
            new_field[i, Ny+1, k] = (field[i, Ny, k]*3-field[i, Ny-1, k])/2
            for j=2:Ny
                new_field[i, j, k] = 0.5 * (field[i, j-1, k] + field[i, j, k])
            end
        end
    elseif dim == 3
        new_field = Array{T}(undef, Nx, Ny, Nz+1)
        for i=1:Nx, j=1:Ny
            new_field[i, j, 1] = (field[i, j, 1]*3-field[i, j, 2])/2
            new_field[i, j, Nz+1] = (field[i, j, Nz]*3-field[i, j, Nz-1])/2
            for k=2:Nz
                new_field[i, j, k] = 0.5 * (field[i, j, k-1] + field[i, j, k])
            end
        end
    else
        throw(ArgumentError("Invalid dimension specified. Please choose between 1, 2, or 3."))
    end

    return new_field
end



function reallocate_uv(field::AbstractArray{T, 2}; dim::Int=1) where T
    Nx, Ny = size(field)

    # Depending on the dimension specified by `dim`, we'll interpolate the data.
    # The interpolated field will have one additional element along the specified dimension.
    if dim == 1
        new_field = Array{T}(undef, Nx+1, Ny)
        for j=1:Ny
            new_field[1, j] = (field[1, j]*3-field[2, j])/2  #simple linear interpolation 
            new_field[Nx+1, j] = (field[Nx, j]*3-field[Nx-1, j])/2
            for i=2:Nx
                new_field[i, j] = 0.5 * (field[i-1, j] + field[i, j])
            end
        end
    elseif dim == 2
        new_field = Array{T}(undef, Nx, Ny+1)
        for i=1:Nx
            new_field[i, 1] = (field[i, 1]*3-field[i, 2])/2 
            new_field[i, Ny+1] = (field[i, Ny]*3-field[i, Ny-1])/2
            for j=2:Ny
                new_field[i, j] = 0.5 * (field[i, j-1] + field[i, j])
            end
        end
    else
        throw(ArgumentError("Invalid dimension specified. Please choose between 1 or 2."))
    end

    return new_field
end
