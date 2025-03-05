#include <cuda_runtime.h>
#include "cloth_simulation.h"

// CUDA Kernel
__global__ void waveSimulationKernel(float4 *pos, float time)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POINTS_TOTAL)
    {
        float x = pos[idx].x;
        float y = pos[idx].y;
        float wave = 0.05f * sinf(2.0f * 3.1415f * x + time);
        pos[idx] = make_float4(x, y + wave, 0.0f, 1.0f);
    }
}

// Function to launch CUDA kernel
extern "C" void launchWaveSimulationKernel(float4 *pos, float time)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (POINTS_TOTAL + threadsPerBlock - 1) / threadsPerBlock;
    waveSimulationKernel<<<blocksPerGrid, threadsPerBlock>>>(pos, time);
    cudaDeviceSynchronize(); // Ensure execution completes
}
