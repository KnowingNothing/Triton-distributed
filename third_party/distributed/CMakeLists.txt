project(Distributed LANGUAGES CXX)

set(DISTRIBUTED_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")

# ============ Add check for includes here (for future) =============

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY DISTRIBUTED_LIBS "")

# ============ Define a function to create object libraries ============
function(add_distributed_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module pybind11::headers)

  target_include_directories(${name}
    PRIVATE
      "${DISTRIBUTED_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "DISTRIBUTED_LIBS"
  set_property(GLOBAL APPEND PROPERTY DISTRIBUTED_LIBS ${name})
endfunction()

# ============ Add subdirectory with actual code that calls add_distributed_library ============
add_subdirectory("${DISTRIBUTED_SRC_DIR}")

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_distributed_obj_libs GLOBAL PROPERTY DISTRIBUTED_LIBS)

if(NOT _distributed_obj_libs)
  message(WARNING "No object libraries were defined in 'DISTRIBUTED_LIBS'!")
endif()

set(_distributed_obj_sources "")
foreach(_lib IN LISTS _distributed_obj_libs)
  list(APPEND _distributed_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(distributed SHARED ${_distributed_obj_sources})

target_link_libraries(distributed PRIVATE Python3::Module)
