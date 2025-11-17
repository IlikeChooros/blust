add_library(blust)

# Add defines - USE_CUDA_BACKEND is defined in the main CMakeLists file as option

add_subdirectory(datasets)

target_sources(blust PRIVATE
	"src/models/Model.cpp"
	"src/cuda/cuda_driver.cpp"
	"src/datasets/mnist.cpp"
	"src/operations/cpu.cpp"
	"src/operations/operations.cpp"
	"src/tensor.cpp"
)

target_compile_definitions(blust PRIVATE ENABLE_CUDA_BACKEND=0)

if(USE_CUDA_BACKEND)
	target_link_libraries(blust PUBLIC CUDA::cuda_driver)
endif()

target_include_directories(blust PUBLIC include)