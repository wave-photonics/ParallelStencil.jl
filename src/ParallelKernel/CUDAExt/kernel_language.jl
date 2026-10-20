## KERNEL LANGUAGE: ASYNCHRONOUS COPIES TO SHARED MEMORY (cp.async)

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

@inline ParallelStencil.ParallelKernel.memcpy_async_commit_cuda() = ccall("llvm.nvvm.cp.async.commit.group", llvmcall, Cvoid, ())
# the wait takes an immediate operand: the number of groups must be a compile-time constant
@inline ParallelStencil.ParallelKernel.memcpy_async_wait_cuda(::Val{N}) where {N} = ccall("llvm.nvvm.cp.async.wait.group", llvmcall, Cvoid, (Int32,), Int32(N))

