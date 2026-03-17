find_path(GLM_INCLUDE_DIR
    NAMES glm/glm.hpp
    PATHS
        /usr/include
        /usr/local/include
        /usr/include/x86_64-linux-gnu
        /opt/local/include
        ${CMAKE_SOURCE_DIR}/external/glm
    DOC "GLM 头文件路径"
)

if(GLM_INCLUDE_DIR)
    set(GLM_FOUND TRUE)
    set(GLM_INCLUDE_DIRS ${GLM_INCLUDE_DIR})
    message(STATUS "找到 GLM: ${GLM_INCLUDE_DIR}")
else()
    set(GLM_FOUND FALSE)
    message(STATUS "未找到 GLM")
endif()

mark_as_advanced(GLM_INCLUDE_DIR)