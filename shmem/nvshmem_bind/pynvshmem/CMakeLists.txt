cmake_minimum_required(VERSION 3.12)

project(pynvshmem LANGUAGES CXX CUDA)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(pybind11 REQUIRED HINTS "${Python3_SITELIB}")

find_package(NVSHMEM REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()


pybind11_add_module(_pynvshmem src/pynvshmem.cc)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")

set_target_properties(_pynvshmem PROPERTIES CXX_STANDARD 17
                                           CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(_pynvshmem PRIVATE nvshmem::nvshmem_host
                                        nvshmem::nvshmem_device)
target_include_directories(_pynvshmem PRIVATE ${NVSHMEM_INCLUDE_DIRS})
target_compile_options(_pynvshmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
