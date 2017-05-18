cmake_minimum_required(VERSION 3.7)
project(sharp)

# Dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)

# Compiler flags
set(CMAKE_CXX_STANDARD 14)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Compile
set(SOURCE_FILES main.cpp sharp.h sharp.cpp Line.cpp Line.h)
add_executable(sharp ${SOURCE_FILES})

# Linking
target_link_libraries( sharp ${OpenCV_LIBS} )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")