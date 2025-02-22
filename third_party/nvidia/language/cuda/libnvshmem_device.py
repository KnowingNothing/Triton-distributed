from triton.language import core
import triton.language as tl
import sys

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))

# class nvshmemi_cmp_type(Enum):
NVSHMEM_CMP_EQ = 0
NVSHMEM_CMP_NE = 1
NVSHMEM_CMP_GT = 2
NVSHMEM_CMP_LE = 3
NVSHMEM_CMP_LT = 4
NVSHMEM_CMP_GE = 5
NVSHMEM_CMP_SENTINEL = sys.maxsize

# class nvshmemi_amo_t(Enum):
NVSHMEMI_AMO_ACK = 1
NVSHMEMI_AMO_INC = 2
NVSHMEMI_AMO_SET = 3
NVSHMEMI_AMO_ADD = 4
NVSHMEMI_AMO_AND = 5
NVSHMEMI_AMO_OR = 6
NVSHMEMI_AMO_XOR = 7
NVSHMEMI_AMO_SIGNAL = 8
NVSHMEM_SIGNAL_SET = 9
NVSHMEM_SIGNAL_ADD = 10
NVSHMEMI_AMO_SIGNAL_SET = NVSHMEM_SIGNAL_SET  # Note - NVSHMEM_SIGNAL_SET == 9
NVSHMEMI_AMO_SIGNAL_ADD = NVSHMEM_SIGNAL_ADD  # Note - NVSHMEM_SIGNAL_ADD == 10
NVSHMEMI_AMO_END_OF_NONFETCH = 11  # end of nonfetch atomics
NVSHMEMI_AMO_FETCH = 12
NVSHMEMI_AMO_FETCH_INC = 13
NVSHMEMI_AMO_FETCH_ADD = 14
NVSHMEMI_AMO_FETCH_AND = 15
NVSHMEMI_AMO_FETCH_OR = 16
NVSHMEMI_AMO_FETCH_XOR = 17
NVSHMEMI_AMO_SWAP = 18
NVSHMEMI_AMO_COMPARE_SWAP = 19
NVSHMEMI_AMO_OP_SENTINEL = sys.maxsize


@core.extern
def my_pe(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _builder=_builder,
    )


@core.extern
def n_pes(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _builder=_builder,
    )


@core.extern
def int_p(dest, value, pe, _builder=None):
    # force have a return value, even not used.
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [dest, value, pe],
        {
            (
                core.pointer_type(core.dtype("int32")),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("nvshmem_int_p", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def remote_ptr(local_ptr, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [local_ptr, pe],
        {(core.pointer_type(core.dtype(core_dtype)), core.dtype(pe_dtype)): (
             "nvshmem_ptr",
             core.pointer_type(core.dtype("int8")),
         )
         for core_dtype in core.dtype.SINT_TYPES + core.dtype.UINT_TYPES + core.dtype.FP_TYPES + core.dtype.OTHER_TYPES
         for pe_dtype in ["int32", "uint32"]},
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def barrier_all(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_barrier_all", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def barrier_all_block(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmemx_barrier_all_block", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def barrier_all_warp(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmemx_barrier_all_warp", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def sync_all(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_sync_all", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def sync_all_block(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmemx_sync_all_block", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def sync_all_warp(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmemx_sync_all_warp", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def quiet(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_quiet", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def fence(_builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_fence", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def getmem_nbi_block(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_getmem_nbi_block",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def getmem_block(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_getmem_block",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def getmem_nbi_warp(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_getmem_nbi_warp",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def getmem_warp(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_getmem_warp",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def getmem_nbi(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmem_getmem_nbi",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def getmem(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmem_getmem",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_block(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_putmem_block",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_nbi_block(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_putmem_nbi_block",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_warp(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_putmem_warp",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_nbi_warp(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmemx_putmem_nbi_warp",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmem_putmem",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_nbi(dest, source, bytes, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, tl.int32): (
                "nvshmem_putmem_nbi",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmem_putmem_signal",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmem_putmem_signal_nbi",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmemx_putmem_signal_block",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmemx_putmem_signal_nbi_block",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmemx_putmem_signal_warp",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pi32_t, _builder=_builder),
            tl.cast(source, tl.pi32_t, _builder=_builder),
            tl.cast(bytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pi32_t, tl.pi32_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmemx_putmem_signal_nbi_warp",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmemx_signal_op",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _builder=None):
    return core.extern_elementwise(
        "libnvshmem_device",
        "",
        [
            sig_addr,
            tl.cast(cmp_, tl.int32, _builder=_builder),
            tl.cast(cmp_val, tl.uint64, _builder=_builder),
        ],  # no cast
        {
            (pi_u64_t, tl.int32, tl.uint64): (
                "nvshmem_signal_wait_until",
                tl.int32,
            ),
        },
        is_pure=False,
        _builder=_builder,
        check_args=False,
    )
