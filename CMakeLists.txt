cmake_minimum_required(VERSION 3.4)

project(BlustDL LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall")
# -O3 -ffast-math -mavx2
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# cuda
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Download gtest
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(googletest)

# Load the CEngine library
add_subdirectory(src)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
install(TARGETS ${PROJECT_NAME})


include(CTest)

# Tests
enable_testing()
add_subdirectory(tests)

