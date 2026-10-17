module ParallelStencil_CUDAExt
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "CUDAExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "CUDAExt", "select_hardware.jl"))
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "CUDAExt", "allocators.jl"))
    include(joinpath(@__DIR__, "..", "src", "ParallelKernel", "CUDAExt", "kernel_language.jl"))
end