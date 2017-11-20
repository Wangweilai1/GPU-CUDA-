#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "String.h"

__global__ void add(int a, int b, int * c)
{
	*c = a + b;
}
int main()
{
	int c;
	int *dev_c = NULL;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&dev_c, sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return -1;
    }
	add<<<1, 1>>>(2, 7, dev_c);
	cudaStatus = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return -1;
    }
	if(dev_c != NULL)
	{
		//printf("2 + 7 = %d\n", *dev_c);
		printf("2 + 7 = %d\n", c);
	}
	cudaFree(dev_c);
	//Second Demo
#if 0
	int count = 0;
	cudaDeviceProp prop;
	memset(&prop, 0x00, sizeof(cudaDeviceProp));
	cudaGetDeviceCount(&count);
	printf("Device Count is %d\n", count);
	for(int i = 0; i <count; ++i)
	{
		cudaGetDeviceProperties(&prop, i);
		printf("Information for Device %d\n", i);
		printf("Name:%s\n", prop.name);
	}
#endif
	//Third Demo. 多GPU环境下选择最优的GPU.
	cudaDeviceProp prop;
	int dev;
	cudaGetDevice(&dev);
	printf("ID of current CUDA device: %d\n", dev);
	memset(&prop, 0x00, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	cudaChooseDevice(&dev, &prop);
	printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
	cudaSetDevice(dev);
    return 0;
}
