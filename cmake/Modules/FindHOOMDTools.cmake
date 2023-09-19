# This requires setuptools-scm to be installed as we will use it to setup the module version
function(set_version target)
    find_package(Python QUIET COMPONENTS Interpreter)
    set(GET_VERSION_SCRIPT "
from setuptools_scm import get_version
print(get_version(), end='')"
    )
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c ${GET_VERSION_SCRIPT}
        ERROR_VARIABLE error
        OUTPUT_VARIABLE GIT_VERSION
        RESULT_VARIABLE exit_code
    )
    if(NOT exit_code EQUAL 0)
        message(FATAL_ERROR
            "The build process depends on setuptools-scm, make sure it is installed. "
            "Got the following error:\n${error}"
        )
    endif()
    target_compile_definitions(${target} PUBLIC GIT_VERSION=${GIT_VERSION})
endfunction()

# Try finding HOOMD first from the current environment
set(HOOMD_GPU_PLATFORM "CUDA" CACHE STRING "GPU backend: CUDA or HIP.")
find_package(HOOMD QUIET)

if(HOOMD_FOUND)
    if(
        ${HOOMD_VERSION} VERSION_GREATER_EQUAL "3.5.0" OR (
            ${HOOMD_VERSION} VERSION_LESS "3" AND
            ${HOOMD_VERSION} VERSION_GREATER_EQUAL "2.6.0"
        )
    )
        message(STATUS "Found HOOMD: ${HOOMD_DIR} (version ${HOOMD_VERSION})")
    else()
        message(FATAL_ERROR
            "Supported HOOMD versions are v2.6.0 to v2.9.7 or >= v3.5.0 "
            "(version ${HOOMD_VERSION})"
        )
    endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_MODULE_PATH})

if(NOT HOOMD_FOUND)
    find_package(HOOMD 2.6.0 REQUIRED)
endif()

# Plugins must be built as shared libraries
if(ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif()

if(${HOOMD_VERSION} VERSION_LESS "3.5.0")
    add_compile_definitions(EXPORT_HALFSTEPHOOK)
    if(${HOOMD_VERSION} VERSION_LESS "3")
        add_compile_definitions(HOOMD2)
    endif()
endif()

if(ENABLE_HIP AND (HIP_PLATFORM STREQUAL "nvcc"))
    add_compile_definitions(ENABLE_CUDA)
endif()

if(NOT HOOMD_INSTALL_PREFIX)
    set(HOOMD_INSTALL_PREFIX ${HOOMD_DIR})
else()
    set(HOOMD_INSTALL_PREFIX "${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")
endif()

if(NOT HOOMD_LIBRARIES)
    set(HOOMD_LIBRARIES HOOMD::_hoomd HOOMD::_md)
endif()
