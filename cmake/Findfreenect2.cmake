find_path(FREENECT2_INCLUDE_DIR
    NAMES libfreenect2/libfreenect2.hpp
    PATHS
        /usr/local/include
        /usr/include
        /usr/local/include/libfreenect2
    DOC "libfreenect2 头文件路径"
)

find_library(FREENECT2_LIBRARY
    NAMES freenect2
    PATHS
        /usr/local/lib
        /usr/lib
        /usr/local/lib64
        /usr/lib/x86_64-linux-gnu
    DOC "libfreenect2 库文件"
)

if(FREENECT2_INCLUDE_DIR AND FREENECT2_LIBRARY)
    set(FREENECT2_FOUND TRUE)
    set(FREENECT2_INCLUDE_DIRS ${FREENECT2_INCLUDE_DIR})
    set(FREENECT2_LIBRARIES ${FREENECT2_LIBRARY})
    message(STATUS "找到 libfreenect2: ${FREENECT2_INCLUDE_DIR}")
else()
    set(FREENECT2_FOUND FALSE)
    message(STATUS "未找到 libfreenect2")
endif()

mark_as_advanced(FREENECT2_INCLUDE_DIR FREENECT2_LIBRARY)