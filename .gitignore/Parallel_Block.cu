
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N (33 * 1024)

__global__ void add(int *a, int *b, int * c)
{
	//threadIdx.x:当前线程的Index. blockIdx:当前线程块的index. blockDim.x:每个线程块中线程的数量.
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < N)
	{
		c[tid] = a[tid] + b[tid];
		//blockDim.x:每个线程块中线程的数量. gridDim.x:线程格中线程块的数量.
		tid += blockDim.x * gridDim.x;
	}
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
	//add<<<a, b>>>其中a表示设备在执行核函数时使用的并行线程块的数量，b表示一个线程块中有b个线程.(其中b不能超过512)
	add<<<128, 128>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
	bool success = true;
	for(int i = 0; i < N; ++i)
	{
		if((a[i] + b[i]) != c[i]){
			printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if(success)
		printf("We did it!\n");
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
