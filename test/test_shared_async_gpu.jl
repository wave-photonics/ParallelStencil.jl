using Test
import ParallelStencil
using ParallelStencil.ParallelKernel
import ParallelStencil.ParallelKernel: @reset_parallel_kernel, @is_initialized, SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU
import ParallelStencil.ParallelKernel: @require
using ParallelStencil.ParallelKernel.Exceptions
# the GPU backends installed in the environment and functional on this machine
TEST_PACKAGES = filter(p -> p in (PKG_CUDA, PKG_AMDGPU) && Base.find_package(string(p)) !== nothing, SUPPORTED_PACKAGES)
@static if PKG_CUDA in TEST_PACKAGES
    import CUDA
    if !CUDA.functional() TEST_PACKAGES = filter!(x->x≠PKG_CUDA, TEST_PACKAGES) end
end
@static if PKG_AMDGPU in TEST_PACKAGES
    import AMDGPU
    if !AMDGPU.functional() TEST_PACKAGES = filter!(x->x≠PKG_AMDGPU, TEST_PACKAGES) end
end
Base.retry_load_extensions()

# Shared memory at offsets and asynchronous copies into it, on the GPU backends: two arrays declared at distinct
# byte offsets of the dynamic shared memory must not alias, and a ring of planes filled with @memcpy_async (some
# lanes zero-filled) must read back exactly, with more than 48 KiB of shared memory per block.
@static for package in TEST_PACKAGES
eval(:(
    @testset "$(basename(@__FILE__)) (package: $(nameof($package)))" begin
        @require !@is_initialized()
        @init_parallel_kernel($package, Float64)
        @require @is_initialized()
        T = Float64

        @testset "@sharedMem at offsets" begin
            TX, TY, NB = 32, 4, 8
            A = @rand(TX * NB, TY)
            B1 = @zeros(TX * NB, TY); B2 = @zeros(TX * NB, TY)
            @inline function twoshared_body!(ix, iy, B1, B2, A, ::Val{TX}, ::Val{TY}) where {TX, TY}
                tx = (@threadIdx()).x; ty = (@threadIdx()).y
                s1 = @sharedMem(eltype(A), (TX, TY))
                s2 = @sharedMem(eltype(A), (TX, TY), TX * TY * sizeof(eltype(A)))
                s1[tx, ty] = A[ix, iy]
                s2[tx, ty] = 2 * A[ix, iy]
                @sync_threads()
                # read the neighbour's value, so that a missing barrier or an alias shows
                tn = tx == TX ? 1 : tx + 1
                B1[ix, iy] = s1[tn, ty]
                B2[ix, iy] = s2[tn, ty]
                return
            end
            @parallel_indices (ix, iy) function twoshared!(B1, B2, A, tx, ty)
                twoshared_body!(ix, iy, B1, B2, A, tx, ty)
                return
            end
            @parallel (NB, 1, 1) (TX, TY, 1) shmem = 2 * TX * TY * sizeof(T) twoshared!(B1, B2, A, Val(TX), Val(TY))
            Ah = Array(A); want = similar(Ah)
            for b in 0:(NB - 1), ty in 1:TY, tx in 1:TX
                want[b * TX + tx, ty] = Ah[b * TX + (tx == TX ? 1 : tx + 1), ty]
            end
            @test Array(B1) == want
            @test Array(B2) == 2 .* want
        end

        @testset "@memcpy_async ring (> 48 KiB, zero-fill)" begin
            N, S, P, NB = 256, 50, 120, 4                 # 256 threads, 50 stages of 256 doubles = 100 KiB, 120 planes
            A = @rand(N, P, NB)
            B = @zeros(N, P, NB)
            @inline function ring_body!(B, A, ::Val{N}, ::Val{S}, ::Val{P}) where {N, S, P}
                t = (@threadIdx()).x; b = (@blockIdx()).x
                r = @sharedMem(eltype(A), (N, S))
                # every 7th lane of every 3rd plane reads nothing and gets zero
                for p in 1:P
                    st = (p - 1) % S + 1
                    ok = !((t % 7 == 0) & (p % 3 == 0))
                    @memcpy_async(r, t + (st - 1) * N, A, t + (p - 1) * N + (b - 1) * N * P, ok)
                    @memcpy_async_commit()
                    if p >= 2
                        @memcpy_async_wait(1)
                        @sync_threads()
                        q = p - 1; sq = (q - 1) % S + 1
                        tn = t == N ? 1 : t + 1
                        B[t, q, b] = r[tn, sq]
                        @sync_threads()
                    end
                end
                @memcpy_async_wait(0)
                @sync_threads()
                tn = t == N ? 1 : t + 1
                B[t, P, b] = r[tn, (P - 1) % S + 1]
                # a copy from a divergent branch (every 32nd lane, as the halo copies of a tiled kernel)
                @sync_threads()
                if t % 32 == 1
                    @memcpy_async(r, t, A, t + N, true)
                end
                @memcpy_async_commit()
                @memcpy_async_wait(0)
                @sync_threads()
                t % 32 == 1 && (B[t, 1, b] = r[t, 1] - A[t + N] + B[t, 1, b])
                return
            end
            @parallel_indices (ix) function ring!(B, A, n, s, p)
                ring_body!(B, A, n, s, p)
                return
            end
            @parallel (NB, 1, 1) (N, 1, 1) shmem = N * S * sizeof(T) ring!(B, A, Val(N), Val(S), Val(P))
            Ah = Array(A); want = zeros(T, N, P, NB)
            for b in 1:NB, p in 1:P, t in 1:N
                tn = t == N ? 1 : t + 1
                want[t, p, b] = ((tn % 7 == 0) & (p % 3 == 0)) ? zero(T) : Ah[tn, p, b]
            end
            @test Array(B) == want
        end
        @reset_parallel_kernel()
    end;
))
end == nothing || true;
