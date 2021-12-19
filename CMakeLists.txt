cmake_minimum_required(VERSION 3.16..3.22)

# Set-up project
project(dlext LANGUAGES C CXX)

include(FindHOOMD.cmake)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

# Plugins must be built as shared libraries
if(ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif()
#
set(BUILD_SHARED_LIBS on)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${HOOMD_ROOT} CACHE PATH "Installation prefix" FORCE)
endif()

set(PYTHON_MODULE_BASE_DIR ${CMAKE_INSTALL_PREFIX})
message(STATUS "Install plugin to: " ${PYTHON_MODULE_BASE_DIR})

# Process subdirectories
add_subdirectory(${PROJECT_NAME})

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
