cmake_minimum_required(VERSION 2.8.3)
project(ROB502_hw4)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/results)
find_path(EIGEN3_INCLUDE_DIR signature_of_eigen3_matrix_library
    /usr/include/eigen3
    /usr/local/include/eigen3
)

if( NOT EIGEN3_INCLUDE_DIR )
     message( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation!")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(ransac)