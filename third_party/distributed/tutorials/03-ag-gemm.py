"""
Fusing Allgather with GEMM
===============

In this tutorial, you will write a simple Allgather GEMM fusion kernel using Dist-Triton.

In doing so, you will learn about:

* Writing a topology-aware allgather kernel with dist-triton communication primitives.

* Writing a GEMM which syronizes and consumes allgathered data.

"""

# %%
# AllGather Kernel is composed of two parts considering the bandwidth gap between intra-node links and inter-node links.
# --------------

import torch


def inter_node_allgather(local_tensor: torch.Tensor, ag_buffer: list[torch.Tensor], signal_buffer: list[torch.Tensor],
                         signal_target, rank, local_world_size, world_size, intranode_ag_stream, internode_ag_stream):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    M_per_rank, N = local_tensor.shape

    # Each rank sends the local_tensor to ranks of other nodes with the same local_rank
    # Assuming there are 2 nodes, each with 4 workers
    # 0-th local tensor ([0] -> [4]), 4-th local tensor ([4] -> [0])
    # 1-th local tensor ([1] -> [5]), 5-th local tensor ([5] -> [1])
    # 2-th local tensor ([2] -> [6]), 6-th local tensor ([6] -> [2])
    # 3-th local tensor ([3] -> [7]), 7-th local tensor ([7] -> [3])
    with torch.cuda.stream(internode_ag_stream):
        grid = lambda META: (int(n_nodes - 1), )
        nvshmem_device_producer_p2p_put_block_kernel[grid](
            ag_buffer[local_rank],
            signal_buffer[local_rank],
            M_per_rank * N,
            local_tensor.element_size(),
            signal_target,
            rank,
            local_world_size,
            world_size,
            num_warps=32,
        )

    # Each rank sends the local_tensor and the received internode tensors to intranode ranks.
    # 0-th and 4-th local tensors ([0]->[1,2,3])
    # 1-th and 5-th local tensors ([1]->[0,2,3])
    # 2-th and 6-th local tensors ([2]->[0,1,3])
    # 3-th and 7-th local tensors ([3]->[0,1,2])
    # 0-th and 4-th local tensors ([4]->[5,6,7])
    # 1-th and 5-th local tensors ([5]->[4,6,7])
    # 2-th and 6-th local tensors ([6]->[4,5,7])
    # 3-th and 7-th local tensors ([7]->[4,5,6])
    with torch.cuda.stream(intranode_ag_stream):
        cp_engine_producer_all_gather_put(local_tensor, ag_buffer, signal_buffer, M_per_rank, N, signal_target, rank,
                                          local_world_size, world_size, intranode_ag_stream)

    intranode_ag_stream.wait_stream(internode_ag_stream)


# %%
# Let's declare a function to perform internode communication.

import triton
import triton.language as tl
import triton.distributed.language as dl
from triton.language.extra import libshmem_device


@triton.jit
def nvshmem_device_producer_p2p_put_block_kernel(
    ag_buffer_ptr,  # *Pointer* to allgather output vector. The rank-th index has been loaded with local tensor
    signal_buffer_ptr,  # *Pointer* to signal barrier.
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size

    for i in range(pid, n_nodes - 1, num_pid):
        # Each SM is assigned to one peer.
        # Peer id is caculated based on pid and local_rank.
        peer = local_rank + (node_rank + i + 1) % n_nodes * local_world_size
        # We use putmem_signal_block to send data and notify the peer.
        # Since this is the allgather operation, the offsets of both src and dst tensor are *rank*.
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,
            ag_buffer_ptr + rank * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


# %%
# Let's also declare a function to perform intranode communication.

from triton.distributed.kernels.nvidia.common_ops import wait_eq, set_signal
from cuda import cudart


def cp_engine_producer_all_gather_put(local_tensor, ag_buffer, signal_buffer, M_per_rank, N, signal_target, rank,
                                      local_world_size, world_size, intranode_ag_stream):
    local_rank = rank % local_world_size
    n_nodes = local_world_size // world_size
    node_rank = rank // local_world_size

    for i in range(1, local_world_size):
        segment = rank * M_per_rank * N
        local_dst_rank = (local_rank + local_world_size - i) % local_world_size
        src_ptr = ag_buffer[local_rank].data_ptr() + segment * local_tensor.element_size()
        dst_ptr = ag_buffer[local_dst_rank].data_ptr() + segment * local_tensor.element_size()
        # Using copy engine to perform intranode transmission
        # Sending rank-th local tensor to other ranks inside the node.
        (err, ) = cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            M_per_rank * N * local_tensor.element_size(),
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            intranode_ag_stream.cuda_stream,
        )
        # Notify the peer that the transmission is done.
        set_signal(signal_buffer[local_dst_rank][rank].data_ptr(), signal_target, intranode_ag_stream, True)

    for i in range(1, n_nodes):
        recv_rank = local_rank + (node_rank + n_nodes - i) % n_nodes * local_world_size
        recv_segment = recv_rank * M_per_rank * N
        # Waiting for the internode data ready
        wait_eq(signal_buffer[local_rank][recv_rank].data_ptr(), signal_target, intranode_ag_stream, True)
        src_ptr = ag_buffer[local_rank].data_ptr() + recv_segment * local_tensor.element_size()
        for j in range(1, local_world_size):
            local_dst_rank = (local_rank + local_world_size - j) % local_world_size
            dst_ptr = ag_buffer[local_dst_rank].data_ptr() + recv_segment * local_tensor.element_size()
            # Sending (local_rank + j*local_world_size) % world_size -th local tensor to other ranks inside the node.
            (err, ) = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                M_per_rank * N * local_tensor.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                intranode_ag_stream.cuda_stream,
            )
            # Notify the peer that the transmission is done.
            set_signal(signal_buffer[local_dst_rank][recv_rank].data_ptr(), signal_target, intranode_ag_stream, True)


# %%
# Now, let's write a GEMM kernel to consume the transfered tensors!
# We use tma to optimize the GEMM performance on the SM90 platform


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_persistent(a_ptr, b_ptr, c_ptr,  #
                                    M, N, K,  #
                                    rank: tl.constexpr, num_ranks: tl.constexpr, ready_ptr, comm_buf_ptr,
                                    BLOCK_SIZE_M: tl.constexpr,  #
                                    BLOCK_SIZE_N: tl.constexpr,  #
                                    BLOCK_SIZE_K: tl.constexpr,  #
                                    GROUP_SIZE_M: tl.constexpr,  #
                                    EPILOGUE_SUBTILE: tl.constexpr,  #
                                    NUM_SMS: tl.constexpr, ready_value: tl.constexpr = 1,
                                    local_world_size: tl.constexpr = 8):  #
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = rank // local_world_size
    nnodes = num_ranks // local_world_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # swizzle m
            if nnodes == 1:
                alpha = 0
                beta = 0
                pid_m = (pid_m + ((((rank ^ alpha) + beta) % num_ranks) * pid_ms_per_rank)) % num_pid_m
            else:
                m_rank = pid_m // pid_ms_per_rank
                pid_m_intra_rank = pid_m - m_rank * pid_ms_per_rank
                m_node_id = m_rank // local_world_size
                m_local_rank = m_rank % local_world_size
                swizzle_m_node_id = (m_node_id + node_id) % nnodes
                swizzle_m_local_rank = (m_local_rank + rank) % local_world_size
                swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank

                pid_m = swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            rank_beg = offs_am // M_per_rank
            rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            # Each tile wait for the corresponding data to ready
            token = dl.wait(ready_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire", waitValue=ready_value)
            a_desc = dl.consume_token(a_desc, token)

        offs_k = ki * BLOCK_SIZE_K
        # Iteration along k-dimension, and performing multiply.
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


# %%
# Now we combine all the kernels here.


def ag_gemm_inter_node_persistent_op(a, b, c, rank, num_ranks, workspace_tensors, barrier_tensors, comm_buf,
                                     ag_stream=None, internode_ag_stream=None, gemm_stream=None, BLOCK_M=128,
                                     BLOCK_N=256, BLOCK_K=64, stages=3, local_world_size=8, signal_target=1):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, K = b.shape

    local_rank = rank % local_world_size
    n_nodes = num_ranks // local_world_size
    num_ag_sms = n_nodes - 1  # only use n_node-1 SMs for internode communication
    num_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count - num_ag_sms

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    gemm_stream = torch.cuda.current_stream() if gemm_stream is None else gemm_stream
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    gemm_stream.wait_stream(current_stream)

    inter_node_allgather(a, workspace_tensors, barrier_tensors, signal_target, rank, local_world_size, num_ranks,
                         ag_stream, internode_ag_stream)

    compiled = None
    with torch.cuda.stream(gemm_stream):

        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        grid = lambda META: (min(
            num_gemm_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
        ), )
        compiled = kernel_consumer_gemm_persistent[grid](
            workspace_tensors[local_rank][:M],
            b,
            c,  #
            M,
            N_per_rank,
            K,  #
            rank,
            num_ranks,
            barrier_tensors[local_rank],
            comm_buf,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            8,
            False,
            NUM_SMS=num_gemm_sms,
            ready_value=signal_target,
            num_stages=stages,
            num_warps=8,
        )

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(gemm_stream)

    return compiled


# We can use a context to wrap all the tensors used at runtime.
# We rely on NVSHMEM to allocate the symmetric memory for communication

import pynvshmem
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class AllGatherGEMMTensorParallelContext:
    rank: int
    num_ranks: int
    num_local_ranks: int
    local_rank: int
    workspace_tensors: List[torch.Tensor]
    barrier_tensors: List[torch.Tensor]
    fake_barrier_tensor: torch.Tensor
    comm_buf: torch.Tensor
    for_correctness: bool = False
    ag_stream: Optional[torch.cuda.streams.Stream] = None
    gemm_stream: Optional[torch.cuda.streams.Stream] = None
    internode_ag_stream: Optional[torch.cuda.streams.Stream] = None
    serial: bool = False
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    stages: int = 3
    autotune: bool = False

    def update(self, rank, num_ranks, num_local_ranks=8, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3,
               for_correctness=False, ag_stream=None, internode_ag_stream=None, gemm_stream=None, serial=False,
               autotune=False):
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_local_ranks = num_local_ranks
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.stages = stages
        self.for_correctness = for_correctness
        self.ag_stream = ag_stream
        self.internode_ag_stream = internode_ag_stream
        self.gemm_stream = gemm_stream
        self.serial = serial
        self.autotune = autotune


def create_ag_gemm_inter_node_context(tensor_A, tensor_B, rank, num_ranks, num_local_ranks=8, max_M=2**14,
                                      max_blocks=65536, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3,
                                      for_correctness=False, ag_stream=None, gemm_stream=None, serial=False,
                                      autotune=False):
    """create context for allgather gemm inter-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        rank (int): current rank
        num_ranks (int): total number of ranks
        max_M: max number of M shape
        max_blocks: max number of blocks on GPU
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        gemm_stream (torch.cuda.streams.Stream, optional): The stream used for gemm, if not provided, use current stream. Defaults to None.
        serial (bool, optional): Make the execution serialized, for debug. Defaults to False.
        autotune (bool, optional): whether to enable autotune. Defaults to False.

    Returns:
        AllGatherGEMMTensorParallelContext
    """
    _, K = tensor_A.shape
    assert tensor_B.shape[
        1] == K, f"tensor_B should has shape (col_major) [N_per_rank, {K}], but get [{tensor_B.shape}]"
    assert tensor_A.dtype == tensor_B.dtype
    dtype = tensor_A.dtype

    local_rank = rank % num_local_ranks

    fake_barrier = torch.ones([num_ranks], dtype=torch.int32, device=tensor_A.device)
    workspaces = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, K], dtype)
    barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], torch.uint64)
    comm_buf = pynvshmem.nvshmem_create_tensor([max_blocks * num_ranks], torch.int32)
    comm_buf.fill_(0)
    barriers[local_rank].fill_(0)
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    ret = AllGatherGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        workspace_tensors=workspaces,
        barrier_tensors=barriers,
        fake_barrier_tensor=fake_barrier,
        comm_buf=comm_buf,
        for_correctness=for_correctness,
        ag_stream=ag_stream,
        internode_ag_stream=torch.cuda.Stream(),
        gemm_stream=gemm_stream,
        serial=serial,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        stages=stages,
        autotune=autotune,
    )

    return ret


# We write a customized kernel to copy the local data to the symmetric memory initially


@triton.jit
def kernel_local_copy_and_barrier_all(
    rank,
    num_ranks,
    local_buf_ptr,
    global_buf_ptr,
    barrier_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = sm_id // num_pid_n
    pid_n = sm_id % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    data_ptr = local_buf_ptr + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
    dst_ptr = global_buf_ptr + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m + (
        pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
    mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)
    mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)

    data = tl.load(data_ptr, mask=mask_data)
    tl.store(dst_ptr, data, mask=mask_dst)


def local_copy_and_barrier_all(rank, num_ranks, local_data, global_data, comm_buf, barrier_ptr, M_per_rank, N,
                               is_internode: bool = False):
    barrier_ptr.fill_(0)
    pynvshmem.nvshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
    grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel_local_copy_and_barrier_all[grid](rank, num_ranks, local_data, global_data, barrier_ptr, M_per_rank, N,
                                            local_data.stride(0), local_data.stride(1), global_data.stride(0),
                                            global_data.stride(1), 128, 256)
    set_signal(barrier_ptr[rank].data_ptr(), 1, torch.cuda.current_stream(), is_internode)


def ag_gemm_inter_node(a, b, ctx=None, rank=None, num_ranks=None, local_world_size=8, signal_target=1):
    """allgather gemm for inter-node
    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        ctx: (Optional[AllGatherGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
    """
    if ctx is None:
        assert rank is not None and num_ranks is not None
        ctx = create_ag_gemm_inter_node_context(a, b, rank, num_ranks)
    M_per_rank, K = a.shape
    N_per_rank, _ = b.shape
    C = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank], dtype=a.dtype, device=a.device)

    # copy local data to the ctx
    local_copy_and_barrier_all(ctx.rank, ctx.num_ranks, a, ctx.workspace_tensors[ctx.local_rank], ctx.comm_buf,
                               ctx.barrier_tensors[ctx.local_rank], M_per_rank, K, is_internode=True)
    # launch the ag_gemm kernel
    ag_gemm_inter_node_persistent_op(a, b, C, ctx.rank, ctx.num_ranks, ctx.workspace_tensors, ctx.barrier_tensors,
                                     ctx.comm_buf, ag_stream=ctx.ag_stream, internode_ag_stream=ctx.internode_ag_stream,
                                     gemm_stream=ctx.gemm_stream, local_world_size=local_world_size,
                                     signal_target=signal_target)

    return C


# %%
# Benchmark
# ---------
#
# We now benchmark the performance of AG+GEMM fused kernel with several popular LLMs on 2 nodes.

from triton.distributed.utils import (initialize_distributed, TP_GROUP, perf_func, dist_print)


# Non-overlap baseline implemented with torch
def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    ag_out: torch.Tensor,
):
    torch.distributed.all_gather_into_tensor(ag_out, local_input, pg)
    ag_gemm_output = torch.matmul(ag_out, local_weight)
    return ag_gemm_output


def perf_test(M, config, pg, local_world_size):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()
    dtype = torch.float16
    device = "cuda"

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert N % world_size == 0
    M_per_rank = M // world_size
    N_per_rank = N // world_size

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda")

    torch_ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")

    # declare ctx
    ctx = create_ag_gemm_inter_node_context(A, B, rank, world_size, max_M=M, BLOCK_M=config["BM"], BLOCK_N=config["BN"],
                                            BLOCK_K=config["BK"], stages=config["stage"], ag_stream=torch.cuda.Stream(),
                                            gemm_stream=torch.cuda.Stream(), autotune=False)

    def triton_func():
        return ag_gemm_inter_node(A, B, ctx=ctx)

    # randomly fill data to check correctness
    for i in range(5):
        A.copy_(torch.randn([M_per_rank, K], dtype=dtype, device=device))
        B.copy_(torch.randn([N_per_rank, K], dtype=dtype, device=device))
        pynvshmem.nvshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        C = triton_func()

    def torch_func():
        return torch_ag_gemm(TP_GROUP, A, B.T, torch_ag_buffer)

    C_golden = torch_func()

    for i in range(world_size):
        torch.distributed.barrier(TP_GROUP)
        if rank == i:
            if not torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3):
                print(f"Rank {rank}")
                print("Golden")
                print(C_golden)
                print("Output")
                print(C)
                print("Max diff", torch.max(torch.abs(C_golden - C)))
                print("Avg diff", torch.mean(torch.abs(C_golden - C)))
                print("Wrong Answer!")

    _, triton_perf = perf_func(triton_func, iters=10, warmup_iters=20)
    _, torch_perf = perf_func(torch_func, iters=10, warmup_iters=20)

    dist_print(
        f"Rank {rank} perf: triton={triton_perf:.2f} ms, torch={torch_perf:.2f} ms, speedup {torch_perf/triton_perf:.2f}",
        need_sync=True, allowed_ranks=list(range(world_size)))


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()

    M = 8192
    config = {"N": 49152, "K": 12288, "BM": 128, "BN": 256, "BK": 64, "stage": 3}
    LOCAL_WORLD_SIZE = 8
    perf_test(M, config, TP_GROUP, local_world_size=LOCAL_WORLD_SIZE)

    torch.distributed.destroy_process_group()
