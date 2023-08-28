function(fetch_dlpack ver)
    CPMAddPackage(NAME dlpack
        VERSION         ${ver}
        GIT_REPOSITORY  https://github.com/dmlc/dlpack.git
        GIT_TAG         v${ver}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   TRUE
    )
    set(dlpack_ADDED ${dlpack_ADDED} PARENT_SCOPE)
    set(dlpack_SOURCE_DIR ${dlpack_SOURCE_DIR} PARENT_SCOPE)
endfunction()

option(FETCH_DLPACK "Fetch DLPack without looking for it locally" OFF)

if(NOT FETCH_DLPACK)
    find_package(dlpack 0.5 QUIET)
endif()

if(dlpack_FOUND)
    message(STATUS "Found dlpack: ${dlpack_DIR} (version ${dlpack_VERSION})")
else()
    fetch_dlpack(0.8)
    if(dlpack_ADDED)
        add_library(dlpack INTERFACE)
        add_library(dlpack::dlpack ALIAS dlpack)
        target_include_directories(dlpack INTERFACE "${dlpack_SOURCE_DIR}/include")
    endif()
endif()
