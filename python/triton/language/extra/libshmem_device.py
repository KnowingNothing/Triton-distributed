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
import sys


def my_pe():
    ...


def n_pes():
    ...


def int_p(dest, value, pe):
    ...


def remote_ptr(local_ptr, pe):
    ...


def barrier_all():
    ...


def barrier_all_block():
    ...


def barrier_all_warp():
    ...


def sync_all():
    ...


def sync_all_block():
    ...


def sync_all_warp():
    ...


def quiet():
    ...


def fence():
    ...


def getmem_nbi_block(dest, source, bytes, pe):
    ...


def getmem_block(dest, source, bytes, pe):
    ...


def getmem_nbi_warp(dest, source, bytes, pe):
    ...


def getmem_warp(dest, source, bytes, pe):
    ...


def getmem_nbi(dest, source, bytes, pe):
    ...


def getmem(dest, source, bytes, pe):
    ...


def putmem_block(dest, source, bytes, pe):
    ...


def putmem_nbi_block(dest, source, bytes, pe):
    ...


def putmem_warp(dest, source, bytes, pe):
    ...


def putmem_nbi_warp(dest, source, bytes, pe):
    ...


def putmem(dest, source, bytes, pe):
    ...


def putmem_nbi(dest, source, bytes, pe):
    ...


def putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def signal_op(sig_addr, signal, sig_op, pe):
    ...


def signal_wait_until(sig_addr, cmp_, cmp_val):
    ...


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
