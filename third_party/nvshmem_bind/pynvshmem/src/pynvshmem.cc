#include <bootstrap_device_host/nvshmem_uniqueid.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <nvshmemx.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

#define CHECK_NVSHMEMX(expr)                                                   \
  do {                                                                         \
    int x = expr;                                                              \
    if (x != NVSHMEMX_SUCCESS) {                                               \
      throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +       \
                               " " #expr " failed with status code " +         \
                               std::to_string(x));                             \
    }                                                                          \
  } while (0)

namespace {
std::array<const char *, 5> kNvshmemInitStatus = {
    "NVSHMEM_STATUS_NOT_INITIALIZED", "NVSHMEM_STATUS_IS_BOOTSTRAPPED",
    "NVSHMEM_STATUS_IS_INITIALIZED", "NVSHMEM_STATUS_LIMITED_MPG",
    "NVSHMEM_STATUS_FULL_MPG"};
void check_nvshmem_init() {
  CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED);
}
} // namespace

#define NVSHMEMI_TYPENAME_P_IMPL_PYBIND(TYPENAME, TYPE)                        \
  void TYPENAME##_p(ptrdiff_t ptr, TYPE value, int peer) {                     \
    check_nvshmem_init();                                                      \
    nvshmem_##TYPENAME##_p((TYPE *)ptr, value, peer);                          \
  }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_IMPL_PYBIND)
#undef NVSHMEMI_TYPENAME_P_IMPL_PYBIND

inline torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                                   c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(
          c10::cuda::current_device());
  auto size =
      torch::elementSize(dtype) *
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  return at::from_blob(
      nvshmem_malloc(size), shape, [](void *ptr) { nvshmem_free(ptr); },
      option_gpu);
}

PYBIND11_MODULE(pynvshmem, m) {
  m.def("nvshmemx_cumodule_init", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_init((CUmodule)module));
  });
  m.def("nvshmemx_cumodule_finalize", [](intptr_t module) {
    CHECK_NVSHMEMX(nvshmemx_cumodule_finalize((CUmodule)module));
  });
  m.def("nvshmem_malloc", [](size_t size) {
    void *ptr = nvshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("nvshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
  m.def("nvshmemx_get_uniqueid", []() {
    nvshmemx_uniqueid_t id;
    CHECK_NVSHMEMX(nvshmemx_get_uniqueid(&id));
    std::string bytes((char *)&id, sizeof(id));
    return pybind11::bytes(bytes);
  });
  m.def("nvshmemx_init_attr_with_uniqueid", [](int rank, int nranks,
                                               pybind11::bytes bytes) {
    nvshmemx_uniqueid_t id;
    std::string id_str = bytes;
    if (id_str.size() != sizeof(id)) {
      throw std::runtime_error(
          "nvshmemx_init_attr_with_uniqueid: invalid size");
    }
    nvshmemx_init_attr_t init_attr;
    CHECK_NVSHMEMX(
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &init_attr));
    memcpy(&id, id_str.data(), sizeof(id));
    CHECK_NVSHMEMX(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &init_attr));
  });
#define NVSHMEMI_TYPENAME_P_PYBIND(TYPENAME, TYPE)                             \
  m.def("nvshmem_" #TYPENAME "_p", &TYPENAME##_p);
  NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_PYBIND)
#undef NVSHMEMI_TYPENAME_P_PYBIND
  m.def("nvshmem_create_tensor",
        [](const std::vector<int64_t> shape, py::object dtype) {
          auto cast_dtype = torch::python::detail::py_object_to_dtype(dtype);
          return create_tensor(shape, cast_dtype);
        });
  m.def("nvshmem_barrier_all", []() {
    check_nvshmem_init();
    nvshmem_barrier_all();
  });
}
