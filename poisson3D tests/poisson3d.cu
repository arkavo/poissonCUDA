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
#define RANGE 40
#define ITER 10000
#define tol 0.001
#define dr 1
#define dt 0.2

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

void display(double* field)
{
    for(int i=0;i<RANGE;i++)
    {
        for(int j=0;j<RANGE;j++)
        {
            for(int k=0;k<RANGE;k++)
            {
                std::cout<<*(field+i+RANGE*j+RANGE*RANGE*k)<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n\n";
    }
}

void DDX(double* R, double* C,int X,int Y,int Z,double dx)
{
    for(int j=1;j<Y-1;j++)
    {
        for(int k=1;k<Z-1;k++)
        {
            for(int i=1;i<X-1;i++)
            {
                *(R+i+ j*X+k*X*Y) = (*(C+1+i+ j*X+k*X*Y) + *(C-1+i+ j*X+k*X*Y) - 2* *(C+i+ j*X+k*X*Y))/(dx*dx);
            }
        }
    }
}
void DDY(double* R, double* C,int X,int Y,int Z, double dy)
{
    for(int j=1;j<Y-1;j++)
    {
        for(int k=1;k<Z-1;k++)
        {
            for(int i=1;i<X-1;i++)
            {
                *(R+i*X+j*X+k*X*Y) = (*(C+X*(i+1)+j*X+k*X*Y) + *(C+X*(i-1)+j*X+k*X*Y) - 2* *(C+i*X+j*X+k*X*Y))/(dy*dy);
            }
        }
    }
}
void DDZ(double* R, double* C,int X,int Y,int Z, double dz)
{
    for(int j=1;j<Y-1;j++)
    {
        for(int k=1;k<Z-1;k++)
        {
            for(int i=1;i<X-1;i++)
            {
                *(R+i*X*Y+j*X+k*X*Y) = (*(C+X*Y*(i+1)+j*X+k*X*Y) + *(C+X*Y*(i-1)+j*X+k*X*Y)- 2* *(C+i*X*Y)+j*X+k*X*Y)/(dz*dz);
            }
        }
    }
}

//Linear cpu time
 int main()
 {   
    unsigned long size = RANGE*RANGE*RANGE;
    //ALL HOST TESTING COPIES COPIUM
    double* field = (double*)malloc(size);
    double* temp_field = (double*)malloc(size);

    //Second Derivatives
    double* ddx = (double*)malloc(size);
    double* ddy = (double*)malloc(size);
    double* ddz = (double*)malloc(size);
    
    double err_max = 10.0;
    int loopctr = 0;
    auto hst_st = std::chrono::high_resolution_clock::now();
    while(abs(err_max)>tol)
    {
    
     for(int i=1;i<RANGE-1;i++)
     {
         for(int j=1;j<RANGE-1;j++)
         {
             for(int k=1;k<RANGE-1;k++)
             {
                *(temp_field+i+RANGE*j+RANGE*RANGE*k) = *(field+i+RANGE*j+RANGE*RANGE*k)+(*(ddx+i+RANGE*j+RANGE*RANGE*k)+*(ddy+i+RANGE*j+RANGE*RANGE*k)+*(ddz+i+RANGE*j+RANGE*RANGE*k))*dt;      
             }
         }
     }
     for(int i=1;i<RANGE-1;i++)
     {
         for(int j=1;j<RANGE-1;j++)
         {
             for(int k=1;k<RANGE-1;k++)
             {
                if(err_max>abs(*(temp_field+i+RANGE*j+RANGE*RANGE*k) - *(field+i+RANGE*j+RANGE*RANGE*k)))
                    err_max = *(temp_field+i+RANGE*j+RANGE*RANGE*k) - *(field+i+RANGE*j+RANGE*RANGE*k);
                loopctr++;
                 //if(loopctr%2000000==0)
                 //    std::cout<<loopctr<<" iters\n";
                *(temp_field+i+RANGE*j+RANGE*RANGE*k) = *(field+i+RANGE*j+RANGE*RANGE*k);
             }
         }
     }
        
    }
    auto hst_en = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = hst_en-hst_st;
    std::cout<<"Duration: "<<duration.count()<<"\n";
    std::cout<<"With "<<loopctr<<" loops\n\n";
    std::cout<<"Error: "<<err_max<<"\n";
    device_list();
    display(field);
}