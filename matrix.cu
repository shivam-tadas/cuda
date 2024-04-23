#include <cuda_runtime.h>
#include<stdio.h>
#include "timer.h"

__global__ void matrix_add(float *a,float *b, float *c, int N,int M){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < N && i < M){
        c[ i * N + j ] = a[i * N + j] + b[i * N + j];
    }
}


int main(){

    cudaDeviceSynchronize();

    const int N = 700;
    const int M = 700;


    float a[N][M],b[N][M],out[N][M],out_p[N][M];

    for (int i = 0; i < N; i++){
        for(int j = 0; j < M;j++){
            a[i][j] = rand();
            b[i][j] = rand();
        }
    }

    const size_t Size = N * M * sizeof(float);

    core::timer cpu_t;
    cpu_t.start();
    for (int i = 0; i < N; i++){
        for(int j = 0; j < M;j++){
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    printf("Cpu time taken :- %f ns\n",cpu_t.nanoseconds());

    // printf("added matrix from cpu\n");
    // for (int i = 0; i < N; i++){
    //     for(int j = 0; j < M;j++){
    //         printf("%f ",out[i][j]);
    //     }
    //     printf("\n");
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
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    core::timer gpu_t;
    gpu_t.start();

    matrix_add<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b,d_out,N,M);

    printf("gpu time taken :- %f ns\n",gpu_t.nanoseconds());

    cudaMemcpy(out_p,d_out,Size,cudaMemcpyDeviceToHost);

    // printf("added matrix from gpu\n");
    // for (int i = 0; i < N; i++){
    //     for(int j = 0; j < M;j++){
    //         printf("%f ",out_p[i][j]);
    //     }
    //     printf("\n");
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    printf("gpu total time taken :- %f ns\n",gpu_total_t.nanoseconds());
    
    return 0;
}