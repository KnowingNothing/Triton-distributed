project(Distributed LANGUAGES CXX)

set(DISTRIBUTED_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============ Add check for includes here (for future) =============

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY DISTRIBUTED_LIBS "")

# ============ Define a function to create object libraries ============
function(add_distributed_library name)
  add_library(${name} SHARED ${ARGN})

  target_link_libraries(${name} PUBLIC Python3::Module pybind11::headers)

  target_include_directories(${name}
    PUBLIC
      "${DISTRIBUTED_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "DISTRIBUTED_LIBS"
  set_property(GLOBAL APPEND PROPERTY DISTRIBUTED_LIBS ${name})
endfunction()

if (USE_TRITON_DISTRIBUTED_AOT)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import triton; from packaging.version import Version; assert Version(triton.__version__) >= Version('3.0.0')"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(TRITON_AOT_GENERATED "triton_aot_generated")
  file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/${TRITON_AOT_GENERATED}")
  set(FLASH_DECODE_KERNEL ${PROJECT_SOURCE_DIR}/distributed/kernels/flash_decode.py:kernel_gqa_fwd_batch_decode_split_kv_persistent)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/distributed/tools/compile_aot.py --workspace ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${TRITON_AOT_GENERATED} --kernels ${FLASH_DECODE_KERNEL}  --library triton_distributed_kernel
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  add_definitions(-DUSE_TRITON_DISTRIBUTED_AOT)
endif()

# ============ Add subdirectory with actual code that calls add_distributed_library ============
add_subdirectory("${DISTRIBUTED_SRC_DIR}")

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_distributed_obj_libs GLOBAL PROPERTY DISTRIBUTED_LIBS)

if(NOT _distributed_obj_libs)
  message(WARNING "No object libraries were defined in 'DISTRIBUTED_LIBS'!")
endif()

