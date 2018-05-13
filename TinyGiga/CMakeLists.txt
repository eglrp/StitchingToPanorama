# CMake entry point.
# Note that this CMakeLists.txt in fully independent from the one in the parent directory
# (since that's the goal of this tutorial !)

# For an introduction to CMake, see
# http://www.cmake.org/cmake/help/cmake_tutorial.html (at least steps 1 and 2)

set(projname "TinyGiga")
project ( ${projname} )

# OpenGL
find_package(OpenGL REQUIRED)
if ( WIN32 )
	set(GLFW_INCLUDE_PATH "E:/libs/OpenGL/include/GLFW")
	set(GLFW_LIBRARY "E:/libs/OpenGL/lib/glfw3.lib" )
	set(GLEW_INCLUDE_PATH "E:/libs/OpenGL/include")
	set(GLEW_LIBRARY_DEBUG debug "E:/libs/OpenGL/lib/glew32d.lib")
	set(GLEW_LIBRARY_RELEASE optimized "E:/libs/OpenGL/lib/glew32.lib")
	set(GLM_INCLUDE_PATH "E:/libs/OpenGL/include")
endif()

set(OPENGL_WARPER_DIR ${PROJECT_SOURCE_DIR}/../OpenGLImageWarper)
include_directories(
    ${GLFW_INCLUDE_PATH}
	${GLEW_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
    ${OPENGL_WARPER_DIR}
    ${OPENGL_WARPER_DIR}/common
)

set(MY_HEADER_FILES
    # C++ header file
    Quad.h
    Mesh.h
    ASAP.h
    FeatureMatch.h
    TinyStitcher.h
    DeepFlow.h
    MeshPyramid.h
    SysUtil.hpp
    ColorCorrect.h
    ${OPENGL_WARPER_DIR}/OpenGLImageWarper.h
    ${OPENGL_WARPER_DIR}/common/shader.hpp
)

set(MY_SOURCE_FILES 
	# C++ code
    Quad.cpp
    Mesh.cpp
    ASAP.cpp
    ASAPadd.cpp
    ASAPwarp.cpp
    FeatureMatch.cpp
    TinyStitcher.cpp
    MeshPyramid.cpp
    DeepFlow.cpp
    ColorCorrect.cpp
    ${OPENGL_WARPER_DIR}/OpenGLImageWarper.cpp
    ${OPENGL_WARPER_DIR}/OpenGLImageWarper.cu
    ${OPENGL_WARPER_DIR}/common/shader.cpp
)

cuda_add_executable(${projname}
	${MY_HEADER_FILES}
	${MY_SOURCE_FILES}
	main.cpp
)

target_link_libraries(${projname}
    ${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
    ${GLFW_LIBRARY}
    ${GLEW_LIBRARY_DEBUG}
    ${GLEW_LIBRARY_RELEASE}
    # OpenCV
    ${OpenCV_LIBS}
    # cuda libs
    ${CUDA_LIBRARY} 	
	${CUDA_npp_LIBRARY} 
	${CUDA_nppc_LIBRARY} 
	${CUDA_nppicc_LIBRARY}
	${CUDA_nppicom_LIBRARY}
	${CUDA_nppidei_LIBRARY}
	${CUDA_nppif_LIBRARY}
	${CUDA_nppig_LIBRARY}
	${CUDA_nppim_LIBRARY}
	${CUDA_nppist_LIBRARY}
	${CUDA_nppisu_LIBRARY}
	${CUDA_nppitc_LIBRARY}
	${CUDA_npps_LIBRARY} 
)
