#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "String.h"
#define N 10

__global__ void add(int *a, int *b, int * c)
{
	int tid = blockIdx.x;
	if(tid < N)
		c[tid] = a[tid] + b[tid];
}
int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMalloc((void**)&dev_c, N*sizeof(int));
	for(int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i*i;
	}
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);
	//N:表示设备在执行核函数时使用的并行线程块的数量，
	add<<<N, 1>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i < N; ++i)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
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
