cmake_minimum_required(VERSION 3.16..3.24)

# Set-up project
project(dlext LANGUAGES C CXX)

# Find HOOMD
set(PROJECT_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
include("${PROJECT_MODULE_PATH}/FindHOOMDTools.cmake")

include(GNUInstallDirs)
include("${PROJECT_MODULE_PATH}/FetchCPM.cmake")
include("${PROJECT_MODULE_PATH}/FetchDLPack.cmake")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "" FORCE)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Create the main library
add_library(${PROJECT_NAME} SHARED "")

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC ${HOOMD_LIBRARIES} dlpack::dlpack)

add_subdirectory(dlext)

# Install
message(STATUS "Plugin will be installed at: ${CMAKE_INSTALL_PREFIX}")

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${HOOMD_INSTALL_PREFIX}
)

install(DIRECTORY include/
    DESTINATION "${HOOMD_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/hoomd/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.h"
)

# Create python module
add_subdirectory(python)
