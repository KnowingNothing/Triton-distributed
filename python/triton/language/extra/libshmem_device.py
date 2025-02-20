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


def getmem_nbi_block(dest, source, bytes, pe):
    ...


def getmem_block(dest, source, bytes, pe):
    ...


def getmem_nbi_warp(dest, source, bytes, pe):
    ...


def getmem_warp(dest, source, bytes, pe):
    ...


def getmem_nbi_thread(dest, source, bytes, pe):
    ...


def getmem_thread(dest, source, bytes, pe):
    ...


def putmem_block(dest, source, bytes, pe):
    ...


def putmem_nbi_block(dest, source, bytes, pe):
    ...


def putmem_warp(dest, source, bytes, pe):
    ...


def putmem_nbi_warp(dest, source, bytes, pe):
    ...


def putmem_thread(dest, source, bytes, pe):
    ...


def putmem_nbi_thread(dest, source, bytes, pe):
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
