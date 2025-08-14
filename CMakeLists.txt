cmake_minimum_required(VERSION 3.16)

project(tensorflow_lite C CXX)   # project name

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

#====================== tensorflow set ======================

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory("${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

#====================== tensorflow set ======================

aux_source_directory(./src/ SRC_LIST)

add_executable(${PROJECT_NAME} 
${SRC_LIST}
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
#    ${CMAKE_CURRENT_LIST_DIR}/include/OpenGL_Tools
)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    OpenCL
    tensorflow-lite
    dl
    pthread
    drm
    GLESv2
    EGL
    
#    VDK   # i.MX8
    glfw  # Ubuntu

)
