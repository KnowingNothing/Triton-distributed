################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call
import sys

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))


def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))


def patch_hash_method_for_pointer_type():
    elem_dtype_list = tl.core.dtype.SINT_TYPES + tl.core.dtype.UINT_TYPES + tl.core.dtype.FP_TYPES + tl.core.dtype.OTHER_TYPES
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(tl.core.pointer_type(tl.core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash


# apply monkey patch in runtime
patch_hash_method_for_pointer_type()

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

# team node
NVSHMEM_TEAM_INVALID = -1
NVSHMEM_TEAM_WORLD = 0
NVSHMEM_TEAM_WORLD_INDEX = 0
NVSHMEM_TEAM_SHARED = 1
NVSHMEM_TEAM_SHARED_INDEX = 1
NVSHMEMX_TEAM_NODE = 2
NVSHMEM_TEAM_NODE_INDEX = 2
NVSHMEMX_TEAM_SAME_MYPE_NODE = 3
NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX = 3
NVSHMEMI_TEAM_SAME_GPU = 4
NVSHMEM_TEAM_SAME_GPU_INDEX = 4
NVSHMEMI_TEAM_GPU_LEADERS = 5
NVSHMEM_TEAM_GPU_LEADERS_INDEX = 5
NVSHMEM_TEAMS_MIN = 6
NVSHMEM_TEAM_INDEX_MAX = sys.maxsize

void_ptr = core.pointer_type(core.void)


@core.extern
def my_pe(_builder=None):
    return extern_call(
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
    return extern_call(
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
def team_my_pe(team, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team],
        {
            (tl.int32, ): ("nvshmem_team_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _builder=_builder,
    )


@core.extern
def team_n_pes(team, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team],
        {
            (tl.int32, ): ("nvshmem_team_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _builder=_builder,
    )


@core.extern
def int_p(dest, value, pe, _builder=None):
    # force have a return value, even not used.
    return extern_call(
        "libnvshmem_device",
        "",
        [dest, value, pe],
        {(
            core.pointer_type(core.dtype("int32")),
            core.dtype("int32"),
            core.dtype("int32"),
        ): ("nvshmem_int_p", ()),  # void return type
         },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def _remote_ptr_wrapper(local_ptr, pe, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [local_ptr, pe],
        {(core.pointer_type(core.void), core.dtype("int32")): (
             "nvshmem_ptr", core.pointer_type(core.void),  # of the same dtype
         )},
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def remote_ptr(local_ptr, pe, _builder=None):
    tl.static_assert(
        local_ptr.dtype.is_ptr(),
        "remote_ptr(local_ptr, pe) local_ptr should be a pointer",
        _builder=_builder,
    )
    tl.static_assert(
        pe.dtype.is_int(),
        "remote_ptr(local_ptr, pe) pe should be an integer",
        _builder=_builder,
    )
    return tl.cast(
        _remote_ptr_wrapper(
            tl.cast(local_ptr, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
            _builder=_builder,
        ),
        local_ptr.dtype,
        _builder=_builder,
    )


@core.extern
def remote_mc_ptr(team, ptr, _builder=None):
    tl.static_assert(ptr.type.is_ptr(), "remote_mc_ptr(team, ptr) should be a pointer", _builder=_builder)
    return extern_call(
        "libnvshmem_device",
        "",
        [tl.cast(team, tl.int32, _builder=_builder),
         tl.cast(ptr, void_ptr, _builder=_builder)],
        {(tl.int32, void_ptr): (
             "nvshmemx_mc_ptr", (ptr.type),  # of the same pointer type like ptr
         )},
        is_pure=True,
        _builder=_builder,
    )


@core.extern
def _barrier_impl(team, SCOPE_SUFFIX: core.constexpr, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team],
        {
            (tl.int32, ): (f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_barrier{SCOPE_SUFFIX.value}", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def barrier(team, _builder=None):
    return _barrier_impl(team, core.constexpr(""), _builder=_builder)


@core.extern
def barrier_block(team, _builder=None):
    return _barrier_impl(team, core.constexpr("_block"), _builder=_builder)


@core.extern
def barrier_warp(team, _builder=None):
    return _barrier_impl(team, core.constexpr("_warp"), _builder=_builder)


@core.extern
def _barrier_all_impl(SCOPE_SUFFIX: core.constexpr, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): (f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_barrier_all{SCOPE_SUFFIX.value}", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def barrier_all(_builder=None):
    return _barrier_all_impl(core.constexpr(""), _builder=_builder)


@core.extern
def barrier_all_block(_builder=None):
    return _barrier_all_impl(core.constexpr("_block"), _builder=_builder)


@core.extern
def barrier_all_warp(_builder=None):
    return _barrier_all_impl(core.constexpr("_warp"), _builder=_builder)


@core.extern
def _sync_all_impl(SCOPE_SUFFIX: core.constexpr, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): (f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_sync_all{SCOPE_SUFFIX.value}", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def sync_all(_builder=None):
    return _sync_all_impl(core.constexpr(""), _builder=_builder)


@core.extern
def sync_all_block(_builder=None):
    return _sync_all_impl(core.constexpr("_block"), _builder=_builder)


@core.extern
def sync_all_warp(_builder=None):
    return _sync_all_impl(core.constexpr("_warp"), _builder=_builder)


@core.extern
def sync(team, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team],
        {
            (tl.int32, ): ("nvshmem_sync", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def _team_sync_impl(team, SCOPE_SUFFIX: core.constexpr, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team],
        {
            (tl.int32, ): (f"nvshmemx_team_sync{SCOPE_SUFFIX.value}", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def team_sync_block(team, _builder=None):
    return _team_sync_impl(team, core.constexpr("_block"), _builder=_builder)


@core.extern
def team_sync_warp(team, _builder=None):
    return _team_sync_impl(team, core.constexpr("_warp"), _builder=_builder)


@core.extern
def quiet(_builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_quiet", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def fence(_builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_fence", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def _getmem_impl(dest, source, nbytes, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""),
                 _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(source, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(nbytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_getmem{NBI.value}{SCOPE_SUFFIX.value}",
                (),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def getmem_nbi_block(dest, source, nbytes, pe, _builder=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_block"), core.constexpr("_nbi"), _builder=_builder)


@core.extern
def getmem_block(dest, source, nbytes, pe, _builder=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_block"), core.constexpr(""), _builder=_builder)


@core.extern
def getmem_nbi_warp(dest, source, nbytes, pe, _builder=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr("_nbi"), _builder=_builder)


@core.extern
def getmem_warp(dest, source, nbytes, pe, _builder=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr(""), _builder=_builder)


@core.extern
def getmem_nbi(dest, source, nbytes, pe, _builder=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr("_nbi"), _builder=_builder)


@core.extern
def getmem(dest, source, nbytes, pe, _builder=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr(""), _builder=_builder)


@core.extern
def _putmem_impl(dest, source, nbytes, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""),
                 _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(source, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(nbytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_putmem{NBI.value}{SCOPE_SUFFIX.value}",
                (),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def putmem_block(dest, source, nbytes, pe, _builder=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_block"), core.constexpr(""), _builder=_builder)


@core.extern
def putmem_nbi_block(dest, source, nbytes, pe, _builder=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_block"), core.constexpr("_nbi"), _builder=_builder)


@core.extern
def putmem_warp(dest, source, nbytes, pe, _builder=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr(""), _builder=_builder)


@core.extern
def putmem_nbi_warp(dest, source, nbytes, pe, _builder=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr("_nbi"), _builder=_builder)


@core.extern
def putmem(dest, source, nbytes, pe, _builder=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr(""), _builder=_builder)


@core.extern
def putmem_nbi(dest, source, nbytes, pe, _builder=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr("_nbi"), _builder=_builder)


@core.extern
def _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, SCOPE_SUFFIX: core.constexpr,
                        NBI: core.constexpr = core.constexpr(""), _builder=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t, "sig_addr should be a pointer of uint64_t", _builder=_builder)
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(source, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(nbytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_putmem_signal{NBI.value}{SCOPE_SUFFIX.value}",
                (),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def putmem_signal(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr(""),
        core.constexpr(""),
        _builder=_builder,
    )


@core.extern
def putmem_signal_nbi(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr(""),
                               core.constexpr("_nbi"), _builder=_builder)


@core.extern
def putmem_signal_block(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_block"),
                               core.constexpr(""), _builder=_builder)


@core.extern
def putmem_signal_nbi_block(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_block"),
                               core.constexpr("_nbi"), _builder=_builder)


@core.extern
def putmem_signal_warp(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_warp"),
                               core.constexpr(""), _builder=_builder)


@core.extern
def putmem_signal_nbi_warp(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_warp"),
                               core.constexpr("_nbi"), _builder=_builder)


@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _builder=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t, "sig_addr should be a pointer of uint64_t", _builder=_builder)
    return extern_call(
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
                (),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _builder=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t, "sig_addr should be a pointer of uint64_t", _builder=_builder)
    return extern_call(
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
                tl.uint64,
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def broadcast(team, dest, source, nelems, pe_root, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team, dest, source, tl.cast(nelems, tl.uint64, _builder=_builder), pe_root],  # no cast
        {
            (tl.int32, tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64, tl.int32):
            ("nvshmem_int8_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int16), tl.pointer_type(tl.int16), tl.uint64, tl.int32):
            ("nvshmem_int16_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int32), tl.pointer_type(tl.int32), tl.uint64, tl.int32):
            ("nvshmem_int32_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int64), tl.pointer_type(tl.int64), tl.uint64, tl.int32):
            ("nvshmem_int64_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint8), tl.pointer_type(tl.uint8), tl.uint64, tl.int32):
            ("nvshmem_uint8_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint16), tl.pointer_type(tl.uint16), tl.uint64, tl.int32):
            ("nvshmem_uint16_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint32), tl.pointer_type(tl.uint32), tl.uint64, tl.int32):
            ("nvshmem_uint32_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint64), tl.pointer_type(tl.uint64), tl.uint64, tl.int32):
            ("nvshmem_uint64_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float16), tl.pointer_type(tl.float16), tl.uint64, tl.int32):
            ("nvshmem_half_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.bfloat16), tl.pointer_type(tl.bfloat16), tl.uint64, tl.int32):
            ("nvshmem_bfloat16_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float32), tl.pointer_type(tl.float32), tl.uint64, tl.int32):
            ("nvshmem_float_broadcast", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float64), tl.pointer_type(tl.float64), tl.uint64, tl.int32):
            ("nvshmem_double_broadcast", (tl.int32)),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def broadcast_warp(team, dest, source, nelems, pe_root, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team, dest, source, tl.cast(nelems, tl.uint64, _builder=_builder), pe_root],  # no cast
        {
            (tl.int32, tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64, tl.int32):
            ("nvshmemx_int8_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int16), tl.pointer_type(tl.int16), tl.uint64, tl.int32):
            ("nvshmemx_int16_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int32), tl.pointer_type(tl.int32), tl.uint64, tl.int32):
            ("nvshmemx_int32_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int64), tl.pointer_type(tl.int64), tl.uint64, tl.int32):
            ("nvshmemx_int64_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint8), tl.pointer_type(tl.uint8), tl.uint64, tl.int32):
            ("nvshmemx_uint8_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint16), tl.pointer_type(tl.uint16), tl.uint64, tl.int32):
            ("nvshmemx_uint16_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint32), tl.pointer_type(tl.uint32), tl.uint64, tl.int32):
            ("nvshmemx_uint32_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint64), tl.pointer_type(tl.uint64), tl.uint64, tl.int32):
            ("nvshmemx_uint64_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float16), tl.pointer_type(tl.float16), tl.uint64, tl.int32):
            ("nvshmemx_half_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.bfloat16), tl.pointer_type(tl.bfloat16), tl.uint64, tl.int32):
            ("nvshmemx_bfloat16_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float32), tl.pointer_type(tl.float32), tl.uint64, tl.int32):
            ("nvshmemx_float_broadcast_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float64), tl.pointer_type(tl.float64), tl.uint64, tl.int32):
            ("nvshmemx_double_broadcast_warp", (tl.int32)),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def broadcast_block(team, dest, source, nelems, pe_root, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team, dest, source, tl.cast(nelems, tl.uint64, _builder=_builder), pe_root],  # no cast
        {
            (tl.int32, tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64, tl.int32):
            ("nvshmemx_int8_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int16), tl.pointer_type(tl.int16), tl.uint64, tl.int32):
            ("nvshmemx_int16_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int32), tl.pointer_type(tl.int32), tl.uint64, tl.int32):
            ("nvshmemx_int32_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int64), tl.pointer_type(tl.int64), tl.uint64, tl.int32):
            ("nvshmemx_int64_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint8), tl.pointer_type(tl.uint8), tl.uint64, tl.int32):
            ("nvshmemx_uint8_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint16), tl.pointer_type(tl.uint16), tl.uint64, tl.int32):
            ("nvshmemx_uint16_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint32), tl.pointer_type(tl.uint32), tl.uint64, tl.int32):
            ("nvshmemx_uint32_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint64), tl.pointer_type(tl.uint64), tl.uint64, tl.int32):
            ("nvshmemx_uint64_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float16), tl.pointer_type(tl.float16), tl.uint64, tl.int32):
            ("nvshmemx_half_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.bfloat16), tl.pointer_type(tl.bfloat16), tl.uint64, tl.int32):
            ("nvshmemx_bfloat16_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float32), tl.pointer_type(tl.float32), tl.uint64, tl.int32):
            ("nvshmemx_float_broadcast_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float64), tl.pointer_type(tl.float64), tl.uint64, tl.int32):
            ("nvshmemx_double_broadcast_block", (tl.int32)),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def broadcastmem_block(team, dest, source, nelems, pe_root, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [
            team,
            tl.cast(dest, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(source, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(nelems, tl.uint64, _builder=_builder),
            pe_root,
        ],  # no cast
        {
            (tl.int32, tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32):
            ("nvshmemx_broadcastmem_block", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def fcollect(team, dest, source, nelems, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team, dest, source, tl.cast(nelems, tl.uint64, _builder=_builder)],  # no cast
        {
            (tl.int32, tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64):
            ("nvshmem_int8_fcollect", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int16), tl.pointer_type(tl.int16), tl.uint64):
            ("nvshmem_int16_fcollect", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int32), tl.pointer_type(tl.int32), tl.uint64):
            ("nvshmem_int32_fcollect", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int64), tl.pointer_type(tl.int64), tl.uint64): ("nvshmem_int64_fcollect",
                                                                                          (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint8), tl.pointer_type(tl.uint8), tl.uint64): ("nvshmem_uint8_fcollect",
                                                                                          (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint16), tl.pointer_type(tl.uint16), tl.uint64): ("nvshmem_uint16_fcollect",
                                                                                            (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint32), tl.pointer_type(tl.uint32), tl.uint64): ("nvshmem_uint32_fcollect",
                                                                                            (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint64), tl.pointer_type(tl.uint64), tl.uint64): ("nvshmem_uint64_fcollect",
                                                                                            (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float16), tl.pointer_type(tl.float16), tl.uint64): ("nvshmem_half_fcollect",
                                                                                              (tl.int32)),
            (tl.int32, tl.pointer_type(tl.bfloat16), tl.pointer_type(tl.bfloat16), tl.uint64):
            ("nvshmem_bfloat16_fcollect", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float32), tl.pointer_type(tl.float32), tl.uint64): ("nvshmem_float_fcollect",
                                                                                              (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float64), tl.pointer_type(tl.float64), tl.uint64): ("nvshmem_double_fcollect",
                                                                                              (tl.int32)),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def fcollect_warp(team, dest, source, nelems, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team, dest, source, tl.cast(nelems, tl.uint64, _builder=_builder)],  # no cast
        {
            (tl.int32, tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64): ("nvshmemx_int8_fcollect_warp",
                                                                                        ()),
            (tl.int32, tl.pointer_type(tl.int16), tl.pointer_type(tl.int16), tl.uint64):
            ("nvshmemx_int16_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int32), tl.pointer_type(tl.int32), tl.uint64):
            ("nvshmemx_int32_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int64), tl.pointer_type(tl.int64), tl.uint64):
            ("nvshmemx_int64_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint8), tl.pointer_type(tl.uint8), tl.uint64):
            ("nvshmemx_uint8_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint16), tl.pointer_type(tl.uint16), tl.uint64):
            ("nvshmemx_uint16_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint32), tl.pointer_type(tl.uint32), tl.uint64):
            ("nvshmemx_uint32_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint64), tl.pointer_type(tl.uint64), tl.uint64):
            ("nvshmemx_uint64_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float16), tl.pointer_type(tl.float16), tl.uint64):
            ("nvshmemx_half_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.bfloat16), tl.pointer_type(tl.bfloat16), tl.uint64):
            ("nvshmemx_bfloat16_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float32), tl.pointer_type(tl.float32), tl.uint64):
            ("nvshmemx_float_fcollect_warp", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float64), tl.pointer_type(tl.float64), tl.uint64):
            ("nvshmemx_double_fcollect_warp", (tl.int32)),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def fcollect_block(team, dest, source, nelems, _builder=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [team, dest, source, tl.cast(nelems, tl.uint64, _builder=_builder)],  # no cast
        {
            (tl.int32, tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64): ("nvshmemx_int8_fcollect_block",
                                                                                        (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int16), tl.pointer_type(tl.int16), tl.uint64):
            ("nvshmemx_int16_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int32), tl.pointer_type(tl.int32), tl.uint64):
            ("nvshmemx_int32_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.int64), tl.pointer_type(tl.int64), tl.uint64):
            ("nvshmemx_int64_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint8), tl.pointer_type(tl.uint8), tl.uint64):
            ("nvshmemx_uint8_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint16), tl.pointer_type(tl.uint16), tl.uint64):
            ("nvshmemx_uint16_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint32), tl.pointer_type(tl.uint32), tl.uint64):
            ("nvshmemx_uint32_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.uint64), tl.pointer_type(tl.uint64), tl.uint64):
            ("nvshmemx_uint64_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float16), tl.pointer_type(tl.float16), tl.uint64):
            ("nvshmemx_half_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.bfloat16), tl.pointer_type(tl.bfloat16), tl.uint64):
            ("nvshmemx_bfloat16_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float32), tl.pointer_type(tl.float32), tl.uint64):
            ("nvshmemx_float_fcollect_block", (tl.int32)),
            (tl.int32, tl.pointer_type(tl.float64), tl.pointer_type(tl.float64), tl.uint64):
            ("nvshmemx_double_fcollect_block", (tl.int32)),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def _putmem_rma_impl(dest, source, nbytes, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""),
                     _builder=None):
    return extern_call(
        "libnvshmemi_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(source, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(nbytes, tl.uint64, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                f"nvshmemi_transfer_rma_put{NBI.value}{SCOPE_SUFFIX.value}",
                (),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def putmem_rma(dest, source, nbytes, pe, _builder=None):
    return _putmem_rma_impl(dest, source, nbytes, pe, core.constexpr(""), _builder=_builder)


@core.extern
def putmem_rma_warp(dest, source, nbytes, pe, _builder=None):
    return _putmem_rma_impl(dest, source, nbytes, pe, core.constexpr("_warp"), _builder=_builder)


@core.extern
def putmem_rma_block(dest, source, nbytes, pe, _builder=None):
    return _putmem_rma_impl(dest, source, nbytes, pe, core.constexpr("_block"), _builder=_builder)


@core.extern
def putmem_rma_nbi(dest, source, nbytes, pe, _builder=None):
    return _putmem_rma_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr("_nbi"), _builder=_builder)


@core.extern
def putmem_rma_nbi_warp(dest, source, nbytes, pe, _builder=None):
    return _putmem_rma_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr("_nbi"),
                            _builder=_builder)


@core.extern
def putmem_rma_nbi_block(dest, source, nbytes, pe, _builder=None):
    return _putmem_rma_impl(dest, source, nbytes, pe, core.constexpr("_block"), core.constexpr("_nbi"),
                            _builder=_builder)


@core.extern
def _putmem_signal_rma_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, SCOPE_SUFFIX: core.constexpr,
                            NBI: core.constexpr = core.constexpr(""), _builder=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t, "sig_addr should be a pointer of uint64_t", _builder=_builder)
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(source, tl.pointer_type(tl.void), _builder=_builder),
            tl.cast(nbytes, tl.uint64, _builder=_builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_builder),
            tl.cast(sig_op, tl.int32, _builder=_builder),
            tl.cast(pe, tl.int32, _builder=_builder),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                # nvshmemi_transfer_put_signal_nbi
                f"nvshmemi_transfer_put_signal{NBI.value}{SCOPE_SUFFIX.value}",
                (),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def putmem_signal_rma(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_rma_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr(""),
                                   _builder=_builder)


@core.extern
def putmem_signal_rma_warp(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_rma_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_warp"),
                                   _builder=_builder)


@core.extern
def putmem_signal_rma_block(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_rma_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_block"),
                                   _builder=_builder)


@core.extern
def putmem_signal_rma_nbi(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_rma_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr(""),
                                   core.constexpr("_nbi"), _builder=_builder)


@core.extern
def putmem_signal_rma_nbi_warp(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_rma_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_warp"),
                                   core.constexpr("_nbi"), _builder=_builder)


@core.extern
def putmem_signal_rma_nbi_block(dest, source, nbytes, sig_addr, signal, sig_op, pe, _builder=None):
    return _putmem_signal_rma_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_block"),
                                   core.constexpr("_nbi"), _builder=_builder)
