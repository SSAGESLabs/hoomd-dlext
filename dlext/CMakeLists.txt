set(COMPONENT_NAME dlpack_extension)

set(
    ${COMPONENT_NAME}_sources
    DLSystemView.cc
    PyDLSystemView.cc
)

pybind11_add_module(${COMPONENT_NAME} SHARED ${${COMPONENT_NAME}_sources} NO_EXTRAS)

if(APPLE)
    set_target_properties(
        ${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path"
    )
else()
    set_target_properties(
        ${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN"
    )
endif()

# link the library to its dependencies
target_link_libraries(${COMPONENT_NAME} PRIVATE ${HOOMD_LIBRARIES})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if(ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(
           ${COMPONENT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
        )
   endif()
   if(MPI_LINK_FLAGS)
       set_target_properties(
           ${COMPONENT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
        )
   endif()
endif()

fix_cudart_rpath(${COMPONENT_NAME})

# install the library
install(
    TARGETS ${COMPONENT_NAME}
    LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PROJECT_NAME}
)

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command(
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT "Copy hoomd/${COMPONENT_NAME}/${file}"
    )
ENDMACRO()

set(
    files
    __init__.py
)

install(FILES ${files} DESTINATION ${PYTHON_MODULE_BASE_DIR}/${PROJECT_NAME})

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})
