#include <nvshmem.h>
#include <nvshmemx.h>

extern "C" {

__device__ int nvshmem_my_pe_wrapper();

__device__ int nvshmem_n_pes_wrapper();

__device__ void nvshmem_int_p_wrapper(int *destination, int mype, int peer);

__device__ void *nvshmem_ptr_wrapper(void *ptr, int peer);

__device__ void nvshmemx_signal_op_wrapper(uint64_t *sig_addr, uint64_t signal,
                                           int sig_op, int pe);
}
