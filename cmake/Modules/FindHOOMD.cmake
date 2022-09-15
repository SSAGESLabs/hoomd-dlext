# FindHOOMD
# ---------
#
# CMake script for finding HOOMD and setting up all needed compile options
# to create and link a plugin library
#
# Variables taken as input to this module:
# HOOMD_ROOT  --  Location to look for HOOMD, if it is not in the python path
#
# Variables defined by this module:
# HOOMD_INCLUDE_DIR  --  Include directories that need to be set to include HOOMD
# HOOMD_LIB          --  Cached var locating the hoomd library to link to
# HOOMD_LIBRARIES    --  Libraries needed to link to to access hoomd (uncached)
# HOOMD_FOUND
#
# Various ENABLE_ flags are translated from hoomd_config.h
# so this plugin build can match the ABI of the installed hoomd
#
# As a convenience (for the intended purpose of this find script),
# all include directories and definitions needed to compile with all the various libs
# (boost, python, winsoc, etc...) are set within this script

function(find_hoomd_with_python)
    find_package(Python QUIET COMPONENTS Interpreter)
    set(FIND_HOOMD_SCRIPT "
from __future__ import print_function;
import os
try:
    import hoomd
    print(os.path.dirname(hoomd.__file__), end='')
except:
    print('', end='')"
    )
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "${FIND_HOOMD_SCRIPT}"
        OUTPUT_VARIABLE HOOMD_ROOT
    )
    set(HOOMD_DIR ${HOOMD_ROOT} PARENT_SCOPE)
endfunction()

if(HOOMD_ROOT)
    set(HOOMD_DIR ${HOOMD_ROOT})
elseif(DEFINED ENV{HOOMD_ROOT})
    set(HOOMD_DIR $ENV{HOOMD_ROOT})
else()
    find_hoomd_with_python()
endif()

find_path(HOOMD_INCLUDE_DIR
    NAMES HOOMDVersion.h
    HINTS ${HOOMD_DIR}/include
)
if(HOOMD_INCLUDE_DIR)
    file(READ "${HOOMD_INCLUDE_DIR}/HOOMDVersion.h" HOOMD_VERSION_HEADER)
    string(REGEX
        MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" HOOMD_VERSION ${HOOMD_VERSION_HEADER}
    )
endif()
mark_as_advanced(HOOMD_FOUND HOOMD_DIR HOOMD_INCLUDE_DIR HOOMD_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HOOMD
    REQUIRED_VARS HOOMD_DIR HOOMD_INCLUDE_DIR HOOMD_VERSION
)

if(HOOMD_FOUND)
    include_directories(${HOOMD_INCLUDE_DIR})
    # Run all of HOOMD's generic lib setup scripts
    set(CMAKE_MODULE_PATH
        ${HOOMD_DIR}
        ${HOOMD_DIR}/CMake/hoomd
        ${HOOMD_DIR}/CMake/thrust
        ${CMAKE_MODULE_PATH}
    )
    # Grab previously-set hoomd configuration
    include(hoomd_cache)
    # Handle user build options
    include(CMake_build_options)
    include(CMake_preprocessor_flags)
    # setup the install directories
    include(CMake_install_options)
    # Find the python executable and libraries
    include(HOOMDPythonSetup)
    # Find CUDA and set it up
    include(HOOMDCUDASetup)
    # Set default CFlags
    include(HOOMDCFlagsSetup)
    # Include some os specific options
    include(HOOMDOSSpecificSetup)
    # Setup common libraries used by all targets in this project
    include(HOOMDCommonLibsSetup)
    # Setup macros
    include(HOOMDMacros)
    # Setup MPI support
    include(HOOMDMPISetup)

    set(HOOMD_LIB ${HOOMD_DIR}/_hoomd${PYTHON_MODULE_EXTENSION})
    set(HOOMD_MD_LIB ${HOOMD_DIR}/md/_md${PYTHON_MODULE_EXTENSION})
    set(HOOMD_DEM_LIB ${HOOMD_DIR}/dem/_dem${PYTHON_MODULE_EXTENSION})
    set(HOOMD_HPMC_LIB ${HOOMD_DIR}/hpmc/_hpmc${PYTHON_MODULE_EXTENSION})
    set(HOOMD_CGCMM_LIB ${HOOMD_DIR}/cgcmm/_cgcmm${PYTHON_MODULE_EXTENSION})
    set(HOOMD_METAL_LIB ${HOOMD_DIR}/metal/_metal${PYTHON_MODULE_EXTENSION})
    set(HOOMD_DEPRECATED_LIB ${HOOMD_DIR}/deprecated/_deprecated${PYTHON_MODULE_EXTENSION})

    set(HOOMD_LIBRARIES ${HOOMD_LIB} ${HOOMD_COMMON_LIBS})
    set(HOOMD_LIBRARIES ${HOOMD_LIB} ${HOOMD_COMMON_LIBS})
endif(HOOMD_FOUND)
