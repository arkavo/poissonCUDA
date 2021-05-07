#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <chrono>
#include <string.h>
#include <fstream>
#include <iostream>
//using namespace std;
#define RANGE 4
#define ITER 10000
#define tol 0.001
#define dr 1
#define dt 0.2
//ALL HOST TESTING COPIES COPIUM
float field[RANGE][RANGE][RANGE];
float temp_field[RANGE][RANGE][RANGE];
//First Derivatives
float dx[RANGE][RANGE][RANGE];
float dy[RANGE][RANGE][RANGE];
float dz[RANGE][RANGE][RANGE];
//Second Derivatives
float ddx[RANGE][RANGE][RANGE];
float ddy[RANGE][RANGE][RANGE];
float ddz[RANGE][RANGE][RANGE];
//DEVICE COPIES TO BE ALLOCATED DYNAMICALLY

int FX = 0;

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
    printf("Total amount of constant memory:         %zu\n", devProp.totalConstMem);
}
void device_list()
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
}

void display()
{
    for(int i=0;i<RANGE;i++)
    {
        for(int j=0;j<RANGE;j++)
        {
            for(int k=0;k<RANGE;k++)
            {
                std::cout<<field[i][j][k]<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n\n";
    }
}

void deriv(int mode)
{
    if(mode==1)
    {
        for(int i=1;i<RANGE-1;i++)
        {
            for(int j=1;j<RANGE-1;j++)
            {
                for(int k=1;k<RANGE-1;k++)
                {
                    dx[i][j][k] = (field[i+1][j][k] - field[i-1][j][k])/(2.0*dr);
                    dx[0][j][k] = dx[1][j][k];
                    dx[RANGE][j][k] = dx[RANGE-1][j][k];
                    
                    dy[i][j][k] = (field[i][j+1][k] - field[i][j-1][k])/(2.0*dr);
                    dy[i][0][k] = dy[i][1][k];
                    dy[i][RANGE][k] = dy[i][RANGE-1][k];

                    dz[i][j][k] = (field[i][j][k+1] - field[i][j][k-1])/(2.0*dr);
                    dz[i][j][0] = dz[i][j][1];
                    dz[i][j][RANGE-1] = dz[i][j][RANGE];
                }
            }
        }
    }

    if(mode==2)
    {
        for(int i=1;i<RANGE-1;i++)
        {
            for(int j=1;j<RANGE-1;j++)
            {
                for(int k=1;k<RANGE-1;k++)
                {
                    ddx[i][j][k] = (dx[i+1][j][k] - dx[i-1][j][k])/(2.0*dr);
                    ddx[0][j][k] = ddx[1][j][k];
                    ddx[RANGE][j][k] = ddx[RANGE-1][j][k];
                    
                    ddy[i][j][k] = (dy[i][j+1][k] - dy[i][j-1][k])/(2.0*dr);
                    ddy[i][0][k] = ddy[i][1][k];
                    ddy[i][RANGE][k] = ddy[i][RANGE-1][k];

                    ddz[i][j][k] = (dz[i][j][k+1] - dz[i][j][k-1])/(2.0*dr);
                    ddz[i][j][0] = ddz[i][j][1];
                    ddz[i][j][RANGE-1] = ddz[i][j][RANGE];
                }
            }
        }
    }
}

void DECLARE(int nDevice)
{
    unsigned long long PARTITION = (sizeof(double)*(RANGE-1)*(RANGE-1)*(RANGE-1) )/nDevice + 1;
    for(int i=0;i<nDevice;i++)
    {
        cudaSetDevice(i);
        double* dx, dy, dz, ddx, ddy, ddz;
        cudaMalloc((void**)&dx, PARTITION);
        cudaMalloc((void**)&dy, PARTITION);
        cudaMalloc((void**)&dz, PARTITION);
        cudaMalloc((void**)&ddx, PARTITION);
        cudaMalloc((void**)&ddy, PARTITION);
        cudaMalloc((void**)&ddz, PARTITION);
    }
}
//SINGLE GPU SINGLE THREAD FORCE METHOD
__global__ void DEVICE_DIFF_FXN(double* result, double* data,double* d,double* dd,int dimension,int order)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if(dimension==1 && order==1)
    {
        if(index%(RANGE-1)!=0)
        {
            d[index] = data[index+1] - data[index];
        }
    }
    if(dimension==2 && order==1)
    {
        if(index%((RANGE-1)*RANGE-1)!=0)
        {
            d[index] = data[index+RANGE] - data[index];
        }
    }
    if(dimension==3 && order==1)
    {
        if(index%((RANGE-1)*(RANGE-1)*RANGE-1)!=0)
        {
            d[index] = data[index+RANGE*RANGE] - data[index];
        }
    }
    if(dimension==1 && order==2)
    {
        if(index%(RANGE-1)!=0)
        {
            dd[index] = d[index+1] - d[index];
        }
    }
    if(dimension==2 && order==2)
    {
        if(index%((RANGE-1)*RANGE-1)!=0)
        {
            dd[index] = d[index+RANGE] - d[index];
        }
    }
    if(dimension==3 && order==2)
    {
        if(index%((RANGE-1)*(RANGE-1)*RANGE-1)!=0)
        {
            dd[index] = d[index+RANGE*RANGE] - d[index];
        }
    }
    if(order==1)
        result = d;
    if(order==2)
        result = dd;
}
__global__ void INITIAL(double* DEVICE_DATA)
{
    for(int i=0;i<RANGE;i++)
    {
        for(int j=0;j<RANGE;j++)
        {
            for(int k=0;k<RANGE;k++)
            {
                if(i==0||i==RANGE-1||j==0||j==RANGE-1||k==0||k==RANGE-1)
                    *(DEVICE_DATA+i+j*RANGE+k*RANGE*RANGE) = 20.;
                else
                    *(DEVICE_DATA+i+j*RANGE+k*RANGE*RANGE) = 0.;
            }
        }
    }
}
__host__ double* UPDATE(double* DEVICE_DATA, double* up, int mode)
{
    if (mode==1)
    {
        for(int i=0;i<RANGE;i++)
            for(int j=0;j<RANGE;j++)
                for(int k=0;k<RANGE;k++)
                    *(DEVICE_DATA+i+j*RANGE+k*RANGE*RANGE) += *(up+i+j*RANGE+k*RANGE*RANGE) * dt;
    }
    if (mode==2)
    {
        for(int i=0;i<RANGE;i++)
            for(int j=0;j<RANGE;j++)
                for(int k=0;k<RANGE;k++)
                    *(DEVICE_DATA+i+j*RANGE+k*RANGE*RANGE) += *(up+i+j*RANGE+k*RANGE*RANGE) * dt*dt/2;
    }
    return DEVICE_DATA;
}
__global__ void UPDATE(double* R,double* S)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    for(int i=0;i<RANGE;i++)
    {
        R[index+i] += S[index+i];
        S[index+i] = 0;
    }
}
__global__ void ERROR_CHECK(double* DEVICE_P, double* DEVICE_C)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x; 
    double err_max = 0.01;
    for(int i=0;i<RANGE;i++)
    {
        if((DEVICE_C[index+i]-DEVICE_P[index+1])<=err_max)
            FX = 1;        
    }
}
int poisson3d(double* DEVICE_P, double* DEVICE_C, double* DEVICE_CX,double* DEVICE_CXX, double* DEVICE_S_CX,double* DEVICE_S_CXX)
{
    int loopctr = 0;
    double error = 10000.0;
    double err_max = 0.01;
    while (FX==0)
    {
        loopctr += 1;
        INITIAL<<<1,1>>>(DEVICE_P);
        DEVICE_DIFF_FXN<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_C,DEVICE_CX,DEVICE_CXX,1,1);
        UPDATE<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_CX);
        DEVICE_DIFF_FXN<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_C,DEVICE_CX,DEVICE_CXX,1,2);
        UPDATE<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_CX);
        DEVICE_DIFF_FXN<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_C,DEVICE_CX,DEVICE_CXX,1,3);
        UPDATE<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_CX);
        DEVICE_DIFF_FXN<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_C,DEVICE_CX,DEVICE_CXX,2,1);
        UPDATE<<<RANGE,RANGE>>>(DEVICE_S_CXX,DEVICE_CXX);
        DEVICE_DIFF_FXN<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_C,DEVICE_CX,DEVICE_CXX,2,2);
        UPDATE<<<RANGE,RANGE>>>(DEVICE_S_CXX,DEVICE_CXX);
        DEVICE_DIFF_FXN<<<RANGE,RANGE>>>(DEVICE_S_CX,DEVICE_C,DEVICE_CX,DEVICE_CXX,2,3);
        UPDATE<<<RANGE,RANGE>>>(DEVICE_S_CXX,DEVICE_CXX);
    }
    return loopctr;
}

int main(int argc, char* argv[])
{
    //boundary conditions + init
    for(int i=0;i<RANGE;i++)
    {
        for(int j=0;j<RANGE;j++)
        {
            for(int k=0;k<RANGE;k++)
            {
                if(i==0||i==RANGE-1||j==0||j==RANGE-1||k==0||k==RANGE-1)
                    field[i][j][k] = 20;
                else
                    field[i][j][k] = 0;
            }
        }
    }
    //GPU COPIES
    double* DEVICE_P;
    double* DEVICE_C;
    double* DEVICE_CX;
    double* DEVICE_S_CX;
    double* DEVICE_CXX;
    double* DEVICE_S_CXX;
    double* trial;
    unsigned long long TOTAL_SIZE = sizeof(double)*RANGE*RANGE*RANGE;
    cudaMalloc((void**)&DEVICE_P,TOTAL_SIZE);
    cudaMalloc((void**)&DEVICE_C,TOTAL_SIZE);
    cudaMalloc((void**)&DEVICE_CX,TOTAL_SIZE);
    cudaMalloc((void**)&DEVICE_S_CX,TOTAL_SIZE);
    cudaMalloc((void**)&DEVICE_CXX,TOTAL_SIZE);
    cudaMalloc((void**)&DEVICE_S_CXX,TOTAL_SIZE);
    trial = (double*)malloc(TOTAL_SIZE);
    cudaMemcpy(&DEVICE_C,field,TOTAL_SIZE,cudaMemcpyHostToDevice);
    
    cudaMemcpy(&trial,&DEVICE_C,TOTAL_SIZE,cudaMemcpyDeviceToHost);
    
    for(int i=0;i<RANGE;i++){
        for(int j=0;j<RANGE;j++){
            for(int k=0;k<RANGE;k++){
                printf("%f \n",*(trial+i+j*RANGE+j*RANGE*RANGE));}}}
    printf("\n");
    //display();
    
    //Linear cpu time
    
    // float err_max = 10.0;
    // int loopctr = 0;
    // auto hst_st = std::chrono::high_resolution_clock::now();
    // while(abs(err_max)>tol)
    // {
    //     deriv(1);
    //     deriv(2);

        
    //     for(int i=0;i<RANGE;i++)
    //     {
    //         for(int j=0;j<RANGE;j++)
    //         {
    //             for(int k=0;k<RANGE;k++)
    //             {
    //                 dx[i][j][k]+=(ddx[i][j][k])*dt;
    //                 dy[i][j][k]+=(ddy[i][j][k])*dt;
    //                 dz[i][j][k]+=(ddz[i][j][k])*dt;
    //             }
    //         }
    //     }
    //     for(int i=1;i<RANGE-1;i++)
    //     {
    //         for(int j=1;j<RANGE-1;j++)
    //         {
    //             for(int k=1;k<RANGE-1;k++)
    //             {
    //                 temp_field[i][j][k] = field[i][j][k]+(dx[i][j][k]+dy[i][j][k]+dz[i][j][k])*dt;      
    //             }
    //         }
    //     }
    //     for(int i=1;i<RANGE-1;i++)
    //     {
    //         for(int j=1;j<RANGE-1;j++)
    //         {
    //             for(int k=1;k<RANGE-1;k++)
    //             {
    //                 err_max = temp_field[i][j][k] - field[i][j][k];
    //                 loopctr++;
    //                 //if(loopctr%2000000==0)
    //                 //    std::cout<<loopctr<<" iters\n";
    //                 field[i][j][k] = temp_field[i][j][k];
    //             }
    //         }
    //     }
        
    // }
    // auto hst_en = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> duration = hst_en-hst_st;
    // std::cout<<"Duration: "<<duration.count()<<"\n";
    //std::cout<<"With "<<loopctr<<" loops\n\n";
    //std::cout<<"Error: "<<err_max<<"\n";
    //device_list();
    //display();
}