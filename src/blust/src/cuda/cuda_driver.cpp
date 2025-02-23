
#include <cuda.h>

START_BLUST_NAMESPACE

#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel64.fatbin"
#endif

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd_kernel;
float* h_A;
float* h_B;
float* h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;


void lanuch_test_kernel()
{
	cuInit(0);
	
	cuDevice = 
}


END_BLUST_NAMESPACE
