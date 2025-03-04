
#include "nvshmem_wrapper.h"

extern "C" {

__device__ int nvshmem_my_pe_wrapper() { return nvshmem_my_pe(); }

__device__ int nvshmem_n_pes_wrapper() { return nvshmem_n_pes(); }

__device__ void nvshmem_int_p_wrapper(int *destination, int mype, int peer) {
  nvshmem_int_p(destination, mype, peer);
}

__device__ void *nvshmem_ptr_wrapper(void *ptr, int peer) {
  return nvshmem_ptr(ptr, peer);
}

__device__ void nvshmemx_signal_op_wrapper(uint64_t *sig_addr, uint64_t signal,
                                           int sig_op, int pe) {
  return nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
}
}
