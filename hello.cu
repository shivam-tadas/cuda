#include <cuda_runtime.h>
#include<stdio.h>
#include "timer.h"

__global__ void vector_add(float *a,float *b, float *c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        c[i] = a[i] + b[i];
    }
}


int main(){

    cudaDeviceSynchronize();

    const int N = 1000000;


    float *a = new float[N];
    float *b = new float[N];
    float *out = new float[N];
    float *out_p = new float[N];

    for (int i = 0; i < N; i++){
        a[i] = rand();
        b[i] = rand();
    }

    const size_t Size = N * sizeof(float);

    core::timer cpu_t;
    cpu_t.start();
    for (int i = 0; i < N; i++){
        out[i] = a[i] + b[i];
    }
    printf("Cpu time taken :- %f ns\n",cpu_t.nanoseconds());

    // printf("array from the cpu \n");
    // for (int i = 0; i < N; i++){
    //     printf("%d ",out[i]);
    // }

    core::timer gpu_total_t;
    gpu_total_t.start();

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, Size);
    cudaMalloc(&d_b, Size);
    cudaMalloc(&d_out, Size);

    cudaMemcpy(d_a,a,Size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,Size,cudaMemcpyHostToDevice);

    //still don't understand this part :(
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    core::timer gpu_t;
    gpu_t.start();

    vector_add<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b,d_out,N);

    printf("gpu time taken :- %f ns\n",gpu_t.nanoseconds());

    cudaMemcpy(out_p,d_out,Size,cudaMemcpyDeviceToHost);

    // printf("array from the gpu \n");
    // for (int i = 0; i < N; i++){
    //     printf("%d ",out_p[i]);
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    printf("gpu time taken :- %f ns\n",gpu_total_t.nanoseconds());

    return 0;
}