add_library(blust)

target_sources(blust PRIVATE
	"src/models/Model.cpp"
	"src/cuda/cuda_driver.cpp"
 "include/blust/cuda/helper_cuda.h" "include/blust/cuda/helper_functions.h")

target_include_directories(blust PUBLIC include)