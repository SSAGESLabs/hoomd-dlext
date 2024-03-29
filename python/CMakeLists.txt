set(pybind11_MODULE_NAME "_api")

pybind11_add_module(${pybind11_MODULE_NAME} MODULE "")

set_version(${pybind11_MODULE_NAME})
target_sources(${pybind11_MODULE_NAME} PRIVATE hoomd_dlext.cc)
target_compile_features(${pybind11_MODULE_NAME} PRIVATE cxx_std_11)
target_link_libraries(${pybind11_MODULE_NAME} PRIVATE ${PROJECT_NAME})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if(ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(
           ${pybind11_MODULE_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
        )
   endif()
   if(MPI_LINK_FLAGS)
       set_target_properties(
           ${pybind11_MODULE_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
        )
   endif()
endif()

if(${HOOMD_VERSION} VERSION_LESS "4")
    fix_cudart_rpath(${pybind11_MODULE_NAME})
endif()

# Install the library
install(TARGETS ${pybind11_MODULE_NAME}
    DESTINATION "${HOOMD_INSTALL_PREFIX}/${PROJECT_NAME}"
)

install(DIRECTORY ${PROJECT_NAME}
    DESTINATION ${HOOMD_INSTALL_PREFIX}
)
