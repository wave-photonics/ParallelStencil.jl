## KERNEL LANGUAGE: ASYNCHRONOUS COPIES TO SHARED MEMORY (cp.async)

# cooperative groups of CUDA.jl (the module moved into CUDACore with CUDA.jl 6)
const CG = isdefined(CUDA, :CG) ? CUDA.CG : CUDA.CUDACore.CG

# one element of a global array into a shared array, in the zero-filling form: with `ok` false nothing is read
# (the source address is clamped to the first element) and the shared element becomes zero
@inline function ParallelStencil.ParallelKernel.memcpy_async_cuda!(dst::CuDeviceArray{T}, di, src::CuDeviceArray{T}, si, ok) where {T}
    d = reinterpret(Core.LLVMPtr{UInt8, CUDA.AS.Shared}, pointer(dst, di))
    s = reinterpret(Core.LLVMPtr{UInt8, CUDA.AS.Global}, pointer(src, ok ? si : one(si)))
    cp_async_zfill(d, s, ok ? Int32(sizeof(T)) : Int32(0), Val(sizeof(T)))
    return nothing
end
@inline cp_async_zfill(d, s, n, ::Val{4})  = ccall("llvm.nvvm.cp.async.ca.shared.global.4.s", llvmcall, Cvoid,
    (Core.LLVMPtr{UInt8, CUDA.AS.Shared}, Core.LLVMPtr{UInt8, CUDA.AS.Global}, Int32), d, s, n)
@inline cp_async_zfill(d, s, n, ::Val{8})  = ccall("llvm.nvvm.cp.async.ca.shared.global.8.s", llvmcall, Cvoid,
    (Core.LLVMPtr{UInt8, CUDA.AS.Shared}, Core.LLVMPtr{UInt8, CUDA.AS.Global}, Int32), d, s, n)
@inline cp_async_zfill(d, s, n, ::Val{16}) = ccall("llvm.nvvm.cp.async.ca.shared.global.16.s", llvmcall, Cvoid,
    (Core.LLVMPtr{UInt8, CUDA.AS.Shared}, Core.LLVMPtr{UInt8, CUDA.AS.Global}, Int32), d, s, n)

@inline ParallelStencil.ParallelKernel.memcpy_async_commit_cuda() = (CG.pipeline_commit(); nothing)
@inline ParallelStencil.ParallelKernel.memcpy_async_wait_cuda(::Val{N}) where {N} = (CG.pipeline_wait_prior(Val(N)); nothing)


## LAUNCH: DYNAMIC SHARED MEMORY ABOVE 48 KiB

# a kernel launched with more than 48 KiB of dynamic shared memory must opt in through its function attribute
function ParallelStencil.ParallelKernel.shmem_optin_cuda!(k, shmem::Integer)
    if shmem > 48 * 1024
        attr = CUDA.attributes(k.fun)
        if attr[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] < shmem
            attr[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
        end
    end
    return k
end
