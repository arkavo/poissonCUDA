#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <chrono>
#include <string.h>
#include <fstream>
#include <iostream>

using namespace std;

#define RANGE 512
//Global Host data
float DATA[RANGE][RANGE][RANGE];

class Vector3
{
    public:
        float x;
        float y;
        float z;

        void set(float x0,float y0, float z0)
        {
            x = x0;
            y = y0;
            z = z0;
        }
        void display()
        {
            cout << "("<<x<<","<<y<<","<<z<<")\n";
        }
};

class GPU_DATASET
{
    public:
        float *GPU_ADDRESS;
        float *HOST_ADDRESS;
        Vector3 DIMENSIONS;
        Vector3 ST_PT;
        int ACTIVE_DEVICE;

        void SET_DEVICE(int nD)
        {
            ACTIVE_DEVICE = nD;
        }

        void SET_GPU_PTR()
        {
            int xDim = (int)DIMENSIONS.x;
            int yDim = (int)DIMENSIONS.y;
            int zDim = (int)DIMENSIONS.z;
            cudaMalloc(&GPU_ADDRESS,sizeof(float)*xDim*yDim*zDim);
            for(int i=ST_PT.x;i<ST_PT.x+DIMENSIONS.x;i++)
            {
                for(int j=ST_PT.y;j<ST_PT.y+DIMENSIONS.y;j++)
                {
                    for(int k=ST_PT.z;k<ST_PT.z+DIMENSIONS.z;k++)
                    {
                       *(GPU_ADDRESS + i + j*xDim + k*yDim*xDim) = *HOST_ADDRESS;
                    }
                }
            }
        }

        void derv_x()
        {

        }
};


__device__ int getGlobalIdx_3D_3D()
{
    int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
    int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    return threadId;
}

void printDevProp(cudaDeviceProp devProp)
{   
    printf("%s\n", devProp.name);
    printf("Major revision number:                   %d\n", devProp.major);
    printf("Minor revision number:                   %d\n", devProp.minor);
    printf("Total global memory:                     %zu", devProp.totalGlobalMem);
    printf("bytes\n");
    printf("Number of multiprocessors:               %d\n", devProp.multiProcessorCount);
    printf("Total amount of shared memory per block: %zu\n",devProp.sharedMemPerBlock);
    printf("Total registers per block:               %d\n", devProp.regsPerBlock);
    printf("Warp size:                               %d\n", devProp.warpSize);
    printf("Maximum memory pitch:                    %zu\n", devProp.memPitch);
    printf("Total amount of constant memory:         %zu\n",   devProp.totalConstMem);
}

void value(int mode, float val)
{
    srand(time(0));
    for(int i=0;i<RANGE;i++)
    {
        for(int j=0;j<RANGE;j++)
        {
            for(int k=0;k<RANGE;k++)
            {
                DATA[i][j][k] = val*(1-mode) + mode*((rand())%100);
            }
        }
    }
}


int main()
{
    //get Devices
    int nDevices;
    /*Hard Set n
    int nDevices = 1;
    */
    cudaGetDeviceCount(&nDevices);
    //Device list and properties
    for (int i = 0; i < nDevices; i++) 
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printDevProp(prop);
        printf("Device Number: %d\n", i);
        printf("Device name: %s\n", prop.name);
        printf("Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    //end Device list
    //return 0;
    void* ADDRESS_LIST;
    ADDRESS_LIST = (void*)malloc(sizeof(int)*nDevices*nDevices*nDevices);
    ADDRESS_LIST = DATA;
    Vector3 test;
    test.set(RANGE,RANGE,RANGE);
    test.display();
    printf("%p\n",ADDRESS_LIST);
}