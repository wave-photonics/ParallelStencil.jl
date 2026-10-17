## KERNEL LANGUAGE: ASYNCHRONOUS COPIES TO LOCAL MEMORY (LDS DMA, gfx9)

# The global → local (LDS) DMA loads of the CDNA GPUs (`global_load_lds_dword`): lane l of a wavefront writes its
# dword at a wave-uniform LDS base + 4·l, so a copy goes through the DMA only for lanes whose local elements follow
# one another in lane order. @memcpy_async gathers such groups: the first pending lane fixes the base (its element
# minus its lane), the pending lanes on that line form a group, and the group takes one DMA instruction (4-byte
# elements) or two (8-byte elements: lane l' loads dword l' & 1 of the element of lane 32·h + l' >> 1); the loop
# repeats until no lane is left. Lanes with `ok` false write their zero at once and take no part; a group whose base
# falls before the array's first element, other element sizes, and 8-byte elements in a partly active wavefront (the
# remap needs the lanes of both dwords of an element; halo copies in divergent branches) copy at once.

@inline lane_id_amdgpu() = ccall("llvm.amdgcn.mbcnt.hi", llvmcall, Int32, (Int32, Int32), Int32(-1),
    ccall("llvm.amdgcn.mbcnt.lo", llvmcall, Int32, (Int32, Int32), Int32(-1), Int32(0)))
@inline lds_dma4!(g::Core.LLVMPtr{UInt8, 1}, l::Core.LLVMPtr{UInt8, 3}) = ccall("llvm.amdgcn.global.load.lds", llvmcall, Cvoid,
    (Core.LLVMPtr{UInt8, 1}, Core.LLVMPtr{UInt8, 3}, Int32, Int32, Int32), g, l, Int32(4), Int32(0), Int32(0))
@inline bperm_amdgpu(lane::Int32, v::Int32) = AMDGPU.Device.bpermute(lane * Int32(4), v)

@inline function ParallelStencil.ParallelKernel.memcpy_async_amdgpu_dma!(dst::AMDGPU.ROCDeviceArray{T}, di, src::AMDGPU.ROCDeviceArray{T}, si, ok) where {T}
    if !(sizeof(T) == 4 || (sizeof(T) == 8 && AMDGPU.Device.ballot(true) == typemax(UInt64)))
        @inbounds dst[di] = ok ? src[si] : zero(T)
        return nothing
    end
    lane = lane_id_amdgpu()
    d = unsafe_trunc(Int32, di)
    ga = UInt64(UInt(pointer(src, ok ? si : one(si))))
    glo = unsafe_trunc(Int32, ga); ghi = unsafe_trunc(Int32, ga >> 32)
    ok || (@inbounds dst[di] = zero(T))
    pending = AMDGPU.Device.ballot(ok)
    while pending != zero(UInt64)
        lead = Int32(trailing_zeros(pending))
        base = bperm_amdgpu(lead, d) - lead
        mine = ok & ((pending >> lane) & one(UInt64) != zero(UInt64)) & (d == base + lane)
        grp = AMDGPU.Device.ballot(mine)
        pending &= ~grp
        if base < one(Int32)
            mine && (@inbounds dst[di] = src[si])
        elseif sizeof(T) == 4
            # lane l writes the dword at base element + l: its own element
            if mine
                lp = reinterpret(Core.LLVMPtr{UInt8, 3}, pointer(dst, base))
                gp = reinterpret(Core.LLVMPtr{UInt8, 1}, ga)
                lds_dma4!(gp, lp)
            end
        else
            for h in Int32(0):Int32(1)
                (grp >> (Int32(32) * h)) & UInt64(0xffffffff) == zero(UInt64) && continue
                s = Int32(32) * h + (lane >> Int32(1))
                take = (grp >> s) & one(UInt64) != zero(UInt64)
                slo = bperm_amdgpu(s, glo); shi = bperm_amdgpu(s, ghi)
                if take
                    a = (UInt64(reinterpret(UInt32, shi)) << 32) | UInt64(reinterpret(UInt32, slo))
                    gp = reinterpret(Core.LLVMPtr{UInt8, 1}, a + UInt64(4) * UInt64(lane & Int32(1)))
                    lp = reinterpret(Core.LLVMPtr{UInt8, 3}, pointer(dst, base + Int32(32) * h))
                    lds_dma4!(gp, lp)
                end
            end
        end
    end
    return nothing
end

# every vector-memory load of the wavefront complete (vmcnt(0); expcnt and lgkmcnt not waited for): the DMA copies are
# in local memory then, and a barrier makes them visible to the workgroup. The count of groups to leave in flight is
# not honoured (the DMA instructions per group vary with the lanes' layout): the copies of the last commit are waited
# for too, so the copies overlap the work done between their issue and the wait.
@inline ParallelStencil.ParallelKernel.memcpy_async_wait_amdgpu_dma() =
    (ccall("llvm.amdgcn.s.waitcnt", llvmcall, Cvoid, (Int32,), Int32(0x0F70)); nothing)
