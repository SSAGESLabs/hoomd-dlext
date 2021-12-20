cmake_minimum_required(VERSION 3.16..3.22)

# Set-up project
project(dlext LANGUAGES C CXX)

# Try finding HOOMD first from the current environment
find_package(HOOMD 2.6.0 QUIET)

set(PROJECT_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${PROJECT_MODULE_PATH})

if(NOT HOOMD_FOUND)
    find_package(HOOMD 2.6.0 REQUIRED)
endif()

if(NOT HOOMD_INSTALL_PREFIX)
    set(HOOMD_INSTALL_PREFIX ${HOOMD_ROOT})
endif()

include("${PROJECT_MODULE_PATH}/FetchCPM.cmake")
include("${PROJECT_MODULE_PATH}/FetchDLPack.cmake")

# Plugins must be built as shared libraries
if(ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "" FORCE)
endif()

message(STATUS "Install plugin to: " ${CMAKE_INSTALL_PREFIX})

# Process subdirectories
add_subdirectory(${PROJECT_NAME})
add_subdirectory(python)
