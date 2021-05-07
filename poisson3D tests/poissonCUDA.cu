#include <stdio.h>

#define X 4
#define Y 4
#define Z 4

__global__ void DX(double* R, double* C,double dx)
{
    index = threadIdx.x*X + blockIdx.x*X*Y;
    for(int i=1;i<X-1;i++)
    {
        R[index] = (C[index+1]-C[index-1])/(dx*2);
    }
}

__global__ void DY(double* R, double* C,double dy)
{
    index = threadIdx.x + blockIdx.x*X*Y;
    for(int i=1;i<Y-1;i++)
    {
        R[index] = (C[index+X]-C[index-X])/(dy*2);
    }
}

__global__ void DZ(double* R, double* C,double dz)
{
    index = threadIdx.x + blockIdx.x*X;
    for(int i=0;i<Z-1<i++)
    {
        R[index] = (C[index+X*Y]-C[index-X*Y])/(dz*2);
    }
}

__global__ void DXX(double* R, double* C,double dx)
{
    index = threadIdx.x*X + blockIdx.x*X*Y;
    R[index] = (C[index+1]+C[index-1]-2*C[index])/(dx*dx);
}

__global__ void DDY(double* R, double* C,double dy)
{
    index = threadIdx.x + blockIdx.x*X*Y;
    for(int i=1;i<Y-1;i++)
    {
        R[index] = (C[index+X]+C[index-X]-2*C[index])/(dy*dy);
    }
}

__global__ void DZ(double* R, double* C,double dz)
{
    index = threadIdx.x + blockIdx.x*X;
    for(int i=0;i<Z-1<i++)
    {
        R[index] = (C[index+X*Y]+C[index-X*Y]-2*C[index])/(dz*dz);
    }
}