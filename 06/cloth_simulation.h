#ifndef CLOTH_SIMULATION_H
#define CLOTH_SIMULATION_H

#include <Windows.h>
#include <gl/GL.h>
#include "vmath.h"
#include "OGL.h"

#ifdef __CUDACC__ // If compiling with NVCC (CUDA compiler)
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    void computeClothCPU(int pointsX, int pointsY);
    void computeClothCUDA(int pointsX, int pointsY);

#ifdef __cplusplus
}
#endif

// Define Vertex structure differently for CPU (C++) and GPU (CUDA)
struct Vertex
{
#ifdef __CUDACC__ // If compiling CUDA
    float4 position;
    float2 texcoord;
#else // If compiling C++
    vmath::vec4 position;
    vmath::vec2 texcoord;
#endif
};

extern Vertex *vertices;
extern GLuint *indices;

#endif // CLOTH_SIMULATION_H
