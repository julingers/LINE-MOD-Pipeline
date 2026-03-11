cmake_minimum_required(VERSION 3.0.0)
project(LINE-MOD VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER gcc-9)
set(CMAKE_CXX_COMPILER g++-9)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

include_directories(${CMAKE_SOURCE_DIR}/include/)
add_definitions(-D GLM_ENABLE_EXPERIMENTAL=TRUE)
# Add source files
file(GLOB_RECURSE SOURCE_FILES 
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cpp)

# Add header files
file(GLOB_RECURSE HEADER_FILES 
	${CMAKE_SOURCE_DIR}/include/*.h
	${CMAKE_SOURCE_DIR}/include/*.hpp)

find_package(SDL2 REQUIRED)
find_package(glm REQUIRED)
find_package(ASSIMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(freenect2 REQUIRED)
find_package(GLEW REQUIRED)
set(OpenCV_DIR /usr/local/lib/cmake/opencv4) # 同时有opencv3和opencv4时需要指定路径
find_package(OpenCV REQUIRED)

find_library(ASSIMP_LIBRARY NAMES ${ASSIMP_LIBRARIES} PATH ${ASSIMP_LIBRARY_DIRS})

function(setup_linemod_target target_name)
	target_include_directories(${target_name} PUBLIC 
			${SDL2_INCLUDE_DIR}
			${glm_INCLUDE_DIRS} # glm只有INCLUDE，没有LIBRARY
			${ASSIMP_INCLUDE_DIR}
			${OPENGL_INCLUDE_DIR}
			${freenect2_INCLUDE_DIR}
			${GLEW_INCLUDE_DIRS}
			${OpenCV_INCLUDE_DIRS}
	)

	target_link_libraries(${target_name} PUBLIC
			SDL2
			${ASSIMP_LIBRARY}
			${OPENGL_gl_LIBRARY}
			freenect2
			${GLEW_LIBRARIES}
			${OpenCV_LIBRARIES}
			stdc++fs # C++17文件系统
    )
endfunction()

add_executable(Detector detector.cpp ${HEADER_FILES} ${SOURCE_FILES})
setup_linemod_target(Detector)

add_executable(Template_Generator templateGeneration.cpp ${HEADER_FILES} ${SOURCE_FILES})
setup_linemod_target(Template_Generator)

add_executable(Arcuo_Benchmark arucoBenchmark.cpp ${HEADER_FILES} ${SOURCE_FILES})
setup_linemod_target(Arcuo_Benchmark)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# 重写了官方宏add_executable并将find_package塞入其中, 会重复查找依赖库，add_executable3次会查找3次
# macro (add_executable _name)
# 	_add_executable(${ARGV})
# 	# SDL2
# 	find_package(SDL2 REQUIRED)
# 	target_link_libraries(${_name}  ${SDL2_LIBRARY})
# 	target_include_directories(${_name}  PUBLIC ${SDL2_INCLUDE_DIR})

# 	# GLM
# 	find_package(glm REQUIRED)
# 	target_link_libraries(${_name}  ${glm_LIBRARY})
# 	target_include_directories(${_name}  PUBLIC ${glm_INCLUDE_DIR})

# 	# Assimp
# 	find_package(ASSIMP REQUIRED)
# 	find_library(ASSIMP_LIBRARY NAMES ${ASSIMP_LIBRARIES} PATH ${ASSIMP_LIBRARY_DIRS})
# 	target_link_libraries(${_name}  ${ASSIMP_LIBRARY})
# 	target_include_directories(${_name}  PUBLIC ${ASSIMP_INCLUDE_DIR})

# 	# OpenGL
# 	find_package(OpenGL REQUIRED)
# 	target_link_libraries(${_name}  ${OPENGL_gl_LIBRARY})
# 	target_include_directories(${_name}  PUBLIC ${OPENGL_INCLUDE_DIR})

# 	#libfreenect2
# 	find_package(freenect2 REQUIRED)
# 	target_link_libraries(${_name}  ${freenect2_LIBRARY})
# 	target_include_directories(${_name}  PUBLIC ${freenect2_INCLUDE_DIR})

# 	# GL Extension Wrangler (GLEW)
# 	find_package(GLEW REQUIRED)
# 	target_link_libraries(${_name}  ${GLEW_LIBRARIES})
# 	target_include_directories(${_name}  PUBLIC ${GLEW_INCLUDE_DIRS})

# 	# OpenCV
# 	set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
# 	find_package(OpenCV REQUIRED)
# 	target_link_libraries(${_name}  ${OpenCV_LIBRARIES})
# 	target_include_directories(${_name}  PUBLIC ${OpenCV_INCLUDE_DIRS})

# 	target_link_libraries(${_name}  stdc++fs)
# 	add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

# 	set_property(TARGET ${_name}  PROPERTY CXX_STANDARD 17)
# 	set_property(TARGET ${_name}  PROPERTY CXX_STANDARD_REQUIRED ON)

# endmacro()


# add_executable(Detector detector.cpp ${HEADER_FILES} ${SOURCE_FILES})
# add_executable(Template_Generator templateGeneration.cpp ${HEADER_FILES} ${SOURCE_FILES})
# add_executable(Arcuo_Benchmark arucoBenchmark.cpp ${HEADER_FILES} ${SOURCE_FILES})

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)