cmake_minimum_required(VERSION 3.7)
project(sharp)

# Dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

find_package(Boost COMPONENTS program_options filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Compiler flags
set(CMAKE_CXX_STANDARD 14)

if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# EasyLogging++ flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELPP_THREAD_SAFE -DELPP_STL_LOGGING")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -DELPP_THREAD_SAFE -DELPP_STL_LOGGING")

# Compile
set(EXTERNAL_FILES external/easyloggingpp_v9.94.2/easylogging++.h external/easyloggingpp_v9.94.2/easylogging++.cc)
set(SOURCE_FILES src/main.cpp include/Sharp.h src/Sharp.cpp src/Line.cpp include/Line.h src/SharpSupport.cpp include/SharpSupport.h test/SharpTest.cpp test/SharpTest.h)
add_executable(sharp ${SOURCE_FILES} ${EXTERNAL_FILES})

# Linking
target_link_libraries(sharp ${OpenCV_LIBS} )
target_link_libraries(sharp ${Boost_LIBRARIES})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")