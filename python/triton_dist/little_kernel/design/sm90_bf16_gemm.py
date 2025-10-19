import little_kernel as lk
import little_kernel.language as ll
from enum import Enum
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda


class GemmType(Enum):
    Normal = 0
    MGroupedContiguous = 1
    MGroupedMasked = 2
    KGroupedContiguous = 3


grouped_layout = None
shape_m = lk.Var("shape_m", ll.uint32)
shape_n = 8192
shape_k = 8192
tensor_map_a = lk.tma.nvgpu.TmaDescriptor()
tensor_map_b = lk.tma.nvgpu.TmaDescriptor()
tensor_map_d = lk.tma.nvgpu.TmaDescriptor()

SHAPE_M = 0
SHAPE_N = shape_n
SHAPE_K = shape_k
kNumGroups = 1
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
kSwizzleDMode = 128
kNumStages = 8
kNumLastStages = ll.cdiv(SHAPE_K, BLOCK_K) % kNumStages
kNumTMAThreads = 128
kNumMathThreads = 128 * (1 if BLOCK_M == 64 else 2)
kNumTMAMulticast = 2
kIsTMAMulticastOnA = True
kNumSMs = 132
kGemmType = GemmType.Normal
cd_dtype = ll.bfloat16

WGMMA = lk.mma.tensor_core.WgMMA_64_X_16_F32BF16BF16_SS(N=BLOCK_N)
Barrier = lk.barrier.nvgpu.ClusterTransactionBarrier()
barrier_dtype = ll.uint64

sync_func = ll.cluster_sync if kNumTMAMulticast > 0 else ll.block_sync

backend = "cuda"


@ll_kernel(backend=backend, is_entry=True)
def sm90_bf16_gemm_impl(
    # runtime inputs
    grouped_layout: ll.Tensor[ll.int32],
    shape_m: ll.int32,
    shape_n: ll.int32,
    shape_k: ll.int32,
    tensor_map_a: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tensor_map_b: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tensor_map_d: ll.const[ll.grid_constant[ll.TmaDescriptor]],
) -> ll.void:
    assert BLOCK_M % WGMMA.M == 0, f"BLOCK_M({BLOCK_M}) must be divisible by WGMMA.M({WGMMA.M})"

    shape_m = shape_m if SHAPE_M == 0 else SHAPE_M
    shape_n, shape_k = shape_n if SHAPE_N == 0 else SHAPE_N, shape_k if SHAPE_K == 0 else SHAPE_K

    # shared memory
    SMEM_D_SIZE = BLOCK_M * BLOCK_N * ll.sizeof(cd_dtype)
    SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * ll.sizeof(ll.bfloat16)
    SMEM_B_SIZE_PER_STAGE = BLOCK_K * BLOCK_N * ll.sizeof(ll.bfloat16)

    # configs
    kFullKOfAllStages = kNumStages * BLOCK_K
    num_iterations = ll.cdiv(shape_k, kFullKOfAllStages)
    warp_idx = ll.__shfl_sync("0xffffffff", ll.thread_x() // 32, 0)
    lane_idx = ll.get_lane_idx()

    # prefetch TMA descriptors
    if (warp_idx == kNumMathThreads // 32 and ll.elect_one_sync()):
        ll.prefetch_tma_descriptor(tensor_map_a)
        ll.prefetch_tma_descriptor(tensor_map_b)
        ll.prefetch_tma_descriptor(tensor_map_d)

    ll.__syncwarp()

    # Align to 1024 bytes for swizzle-128B
    ll.align_memory(1024, scope="dynamic_shared")
    assert SMEM_D_SIZE % 1024 == 0, f"SMEM_D_SIZE({SMEM_D_SIZE}) must be divisible by 1024"

    # Shared memory for tensors
    D_smem_tensor = ll.empty([BLOCK_M, BLOCK_N],
                             dtype=cd_dtype,
                             scope="dynamic_shared")
    A_smem_tensors = [
        ll.empty([BLOCK_M, BLOCK_K], dtype=ll.bfloat16, scope="dynamic_shared")
        for _ in range(kNumStages)
    ]
    assert ll.typeof(A_smem_tensors) == ll.Tensor[ll.Tensor[ll.bfloat16]]
    B_smem_tensors = [
        ll.empty([BLOCK_K, BLOCK_N], dtype=ll.bfloat16, scope="dynamic_shared")
        for _ in range(kNumStages)
    ]

    # Shared memory for barriers
    full_barriers = [
        ll.empty([1], dtype=barrier_dtype, scope="dynamic_shared")
        for _ in range(kNumStages)
    ]
    empty_barriers = [
        ll.empty([1], dtype=barrier_dtype, scope="dynamic_shared")
        for _ in range(kNumStages)
    ]

    # initialize barriers
    if (warp_idx == kNumMathThreads // 32 + 1 and ll.elect_one_sync()):
        for i in ll.unroll(range(kNumStages)):
            ll.init_smem_barrier(full_barriers[i], 1)
            ll.init_smem_barrier(empty_barriers[i],
                                 kNumTMAMulticast * kNumMathThreads // 32)
        ll.fence_smem_barrier_init()

    sync_func()


passes = PASSES[backend]
code = sm90_bf16_gemm_impl.compile(passes, codegen_cuda)
print(code)
