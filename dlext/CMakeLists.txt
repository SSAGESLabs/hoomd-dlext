add_subdirectory(src)

target_include_directories(${PROJECT_NAME} PUBLIC include)

install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hoomd/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.h"
)
