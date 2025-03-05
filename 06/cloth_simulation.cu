#include "cloth_simulation.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel
__global__ void clothKernel(Vertex *vertices, int pointsX, int pointsY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= pointsX || y >= pointsY)
        return;

    int index = y * pointsX + x;
    vertices[index].position = make_float4(x, y, 0.0f, 1.0f);
    vertices[index].texcoord = make_float2(x, y);
}

void computeClothCUDA(int pointsX, int pointsY)
{
    Vertex *d_vertices;
    cudaError_t err = cudaMalloc(&d_vertices, pointsX * pointsY * sizeof(Vertex));
    if (err != cudaSuccess)
    {
        fprintf(gpFILE, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((pointsX + blockSize.x - 1) / blockSize.x, (pointsY + blockSize.y - 1) / blockSize.y);

    clothKernel<<<gridSize, blockSize>>>(d_vertices, pointsX, pointsY);
    cudaDeviceSynchronize();

    cudaFree(d_vertices);
}
