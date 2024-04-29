#include <cuda_runtime.h>
#include <stdio.h>
#include "timer.h"


__global__ void search(int *a, int N, int value, int *found_gpu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (a[i] == value && i < N) {
        atomicAdd(found_gpu, 1); 
    }
}

int main() {
    cudaDeviceSynchronize();

    const int N = 100000000;
    const int value = 7;

    int *a = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 10 + 1;
    }

    const size_t Size = N * sizeof(int);
    int found_cpu = 0;
    core::timer cpu_t;
    cpu_t.start();
    for (int i = 0; i < N; i++) {
        if (a[i] == value) {
            found_cpu++;
        }
    }

    printf("Cpu time taken :- %f sec\n", cpu_t.nanoseconds()/1000000000);

    printf("Number of times value is found by cpu %d \n", found_cpu);

    core::timer gpu_total_t;
    gpu_total_t.start();

    int *d_a;
    cudaMalloc(&d_a, Size);

    cudaMemcpy(d_a, a, Size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    core::timer gpu_t;
    gpu_t.start();

    int *found_gpu = (int*)malloc(sizeof(int));
    *found_gpu = 0;

    search<<<blocksPerGrid, threadsPerBlock>>>(d_a, N, value, found_gpu);

    cudaDeviceSynchronize();  

    printf("gpu time taken :- %f sec\n", gpu_t.nanoseconds()/1000000000);

    cudaMemcpy(&found_cpu, found_gpu, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Number of times value is found by gpu %d \n", found_cpu);

    cudaFree(d_a);
    free(found_gpu);

    printf("gpu total time taken :- %f sec\n", gpu_total_t.nanoseconds()/1000000000);

    return 0;
}
