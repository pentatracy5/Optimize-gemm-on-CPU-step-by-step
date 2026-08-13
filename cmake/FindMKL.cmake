# FindMKL.cmake
#
# Finds Intel oneAPI/Parallel Studio MKL and creates the imported target MKL::MKL.
# The preferred linking model is the single dynamic library (mkl_rt) with the
# Intel OpenMP companion runtime (iomp5).
#
# This module looks in the following places (in order):
#   - The CMake/user variables MKL_ROOT / MKLROOT / ONEAPI_ROOT
#   - The environment variables MKL_ROOT / MKLROOT / ONEAPI_ROOT
#   - Common oneAPI and legacy Intel MKL install locations
#
# The following variables are defined after a successful find:
#   MKL_FOUND          - TRUE if MKL was found
#   MKL_INCLUDE_DIR    - Directory containing mkl.h
#   MKL_RT_LIBRARY     - The mkl_rt single dynamic library
#   MKL_IOMP5_LIBRARY  - The Intel OpenMP runtime (may be empty on some platforms)
#   MKL_LIBRARIES      - List of libraries needed to use MKL
#
# Imported target:
#   MKL::MKL           - UNKNOWN imported target that links mkl_rt and iomp5
#
# If MKL is not found, set MKL_ROOT to the top-level oneAPI MKL directory, e.g.:
#   Windows: C:/Program Files (x86)/Intel/oneAPI/mkl/latest
#   Linux:   /opt/intel/oneapi/mkl/latest

include(FindPackageHandleStandardArgs)

# ---------------------------------------------------------------------------
# Helper: collect candidate MKL installation roots
# ---------------------------------------------------------------------------
set(_MKL_HINTS
    "${MKL_ROOT}"
    "${MKLROOT}"
    "$ENV{MKL_ROOT}"
    "$ENV{MKLROOT}"
    "$ENV{ONEAPI_ROOT}/mkl/latest"
)

# Common default install locations
if(WIN32)
    list(APPEND _MKL_HINTS
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
        "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl"
    )
else()
    list(APPEND _MKL_HINTS
        "/opt/intel/oneapi/mkl/latest"
        "/opt/intel/mkl"
    )
endif()

list(REMOVE_DUPLICATES _MKL_HINTS)

# ---------------------------------------------------------------------------
# Find MKL headers and the single dynamic library
# ---------------------------------------------------------------------------
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS ${_MKL_HINTS}
    PATH_SUFFIXES include
    DOC "Intel MKL include directory"
)

set(_MKL_LIB_PATH_SUFFIXES
    lib
    lib/intel64
    lib/intel64_lin
    lib/intel64_win
)

find_library(MKL_RT_LIBRARY
    NAMES mkl_rt
    HINTS ${_MKL_HINTS}
    PATH_SUFFIXES ${_MKL_LIB_PATH_SUFFIXES}
    DOC "Intel MKL single dynamic library"
)

# ---------------------------------------------------------------------------
# Try to locate the Intel OpenMP runtime (iomp5) that mkl_rt needs.
# In oneAPI the compiler runtime lives next to the MKL folder:
#   <oneapi>/compiler/latest/lib/...
# ---------------------------------------------------------------------------
set(_IOMP_HINTS)
foreach(_root IN LISTS _MKL_HINTS)
    if(NOT _root)
        continue()
    endif()
    get_filename_component(_parent "${_root}" DIRECTORY)
    get_filename_component(_grandparent "${_parent}" DIRECTORY)
    # oneAPI layout: MKL is at <oneapi>/mkl[/latest|/<version>],
    # compiler runtime at <oneapi>/compiler/latest
    list(APPEND _IOMP_HINTS "${_grandparent}/compiler/latest")
    # Older layout: MKL at <base>/mkl, compiler at <base>/compiler
    list(APPEND _IOMP_HINTS "${_grandparent}/compiler")
endforeach()
list(REMOVE_DUPLICATES _IOMP_HINTS)

if(WIN32)
    find_library(MKL_IOMP5_LIBRARY
        NAMES libiomp5md iomp5
        HINTS ${_IOMP_HINTS}
        PATH_SUFFIXES lib lib/intel64 lib/intel64_win
        DOC "Intel OpenMP runtime library"
    )
else()
    find_library(MKL_IOMP5_LIBRARY
        NAMES iomp5 libiomp5
        HINTS ${_IOMP_HINTS}
        PATH_SUFFIXES lib lib/intel64 lib/intel64_lin
        DOC "Intel OpenMP runtime library"
    )
endif()

# ---------------------------------------------------------------------------
# Standard CMake find-package handling
# ---------------------------------------------------------------------------
find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE_DIR MKL_RT_LIBRARY
)

if(MKL_FOUND AND NOT TARGET MKL::MKL)
    add_library(MKL::MKL UNKNOWN IMPORTED)
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
        IMPORTED_LOCATION "${MKL_RT_LIBRARY}"
    )
    if(MKL_IOMP5_LIBRARY)
        set_property(TARGET MKL::MKL APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES "${MKL_IOMP5_LIBRARY}"
        )
    endif()
endif()

set(MKL_LIBRARIES "${MKL_RT_LIBRARY}")
if(MKL_IOMP5_LIBRARY)
    list(APPEND MKL_LIBRARIES "${MKL_IOMP5_LIBRARY}")
endif()

mark_as_advanced(MKL_INCLUDE_DIR MKL_RT_LIBRARY MKL_IOMP5_LIBRARY)
