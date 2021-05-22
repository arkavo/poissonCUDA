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

#define RANGE 4
#define ITER 10000

#define dr 1
#define dt 0.2

//DEVICE COPIES TO BE ALLOCATED DYNAMICALLY


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

void DDX(double *R, double *C,int X,int Y,int Z,double dx)
{
    for(int j=1;j<Y-1;j++)
    {
        for(int k=1;k<Z-1;k++)
        {
            for(int i=1;i<X-1;i++)
            {
                *(R + i + j*X + k*X*Y) = (*(C + 1 + i + j*X + k*X*Y) + *(C-1 + i + j*X + k*X*Y) - 2* *(C + i + j*X + k*X*Y))/(dx);
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
                *(R+i*X+j*X+k*X*Y) = (*(C+X*(i+1)+j*X+k*X*Y) + *(C+X*(i-1)+j*X+k*X*Y) - 2* *(C+i*X+j*X+k*X*Y))/(dy);
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
                *(R+i*X*Y+j*X+k*X*Y) = (*(C+X*Y*(i+1)+j*X+k*X*Y) + *(C+X*Y*(i-1)+j*X+k*X*Y)- 2* *(C+i*X*Y)+j*X+k*X*Y)/(dz);
            }
        }
    }
}
void ADD(double* field,double* temp,int X,int Y,int Z,double dtt)
{
    for(int i=1;i<X-1;i++)
    {
        for(int j=1;j<Y-1;j++)
        {
            for(int k=1;k<Z-1;k++)
            {
                *(field + i + Y*j + Y*Z*k) += *(temp + i + Y*j + Y*Z*k) * dtt;
            }
        }
    }
}
void set(double* field)
{
    for(int i=0;i<RANGE;i++)
    {
        for(int j=0;j<RANGE;j++)
        {
            for(int k=0;k<RANGE;k++)
            {
                *(field+i+RANGE*j+RANGE*RANGE*k) = i+j+k;
            }
        }
    }
}

//Linear cpu time
 int main()
 {   
    unsigned long size = sizeof(double)*RANGE*RANGE*RANGE;
    //ALL HOST TESTING COPIES COPIUM
    double tol = 0.002;
    std::cout<<size<<"\n";
    std::cout<<"Part 1\n";
    double *field = (double*)malloc(size);
    double *temp_field = (double*)malloc(size);
    set(field);
    set(temp_field);
    //Second Derivatives
    double *ddx = (double*)malloc(size);
    double *ddy = (double*)malloc(size);
    double *ddz = (double*)malloc(size);
    std::cout<<"Part 2\n";
    double err_max = 10.0;
    double cur_err_max = 0.0;
    int loopctr = 0;
    auto hst_st = std::chrono::high_resolution_clock::now();
    std::cout<<"Part 3\n";

    display(field);
    while(err_max>tol)
    {
        cur_err_max = 0.;
        loopctr++;
        DDX(ddx,field,RANGE,RANGE,RANGE,4.);
        DDY(ddy,field,RANGE,RANGE,RANGE,4.);
        DDZ(ddz,field,RANGE,RANGE,RANGE,4.);
        std::cout<<"Part 4\n";
        ADD(temp_field,ddx,RANGE,RANGE,RANGE,0.4);
        ADD(temp_field,ddx,RANGE,RANGE,RANGE,0.4);
        ADD(temp_field,ddx,RANGE,RANGE,RANGE,0.4);
        
        display(field);
        std::cout<<"Part 5\n";
        for(int i=1;i<RANGE-1;i++)
        {
            for(int j=1;j<RANGE-1;j++)
            {
                for(int k=1;k<RANGE-1;k++)
                {
                    if(cur_err_max<abs(*(temp_field+i+RANGE*j+RANGE*RANGE*k) - *(field+i+RANGE*j+RANGE*RANGE*k)))
                    {
                        cur_err_max = abs(*(temp_field+i+RANGE*j+RANGE*RANGE*k) - *(field+i+RANGE*j+RANGE*RANGE*k));
                        *(temp_field+i+RANGE*j+RANGE*RANGE*k) = *(field+i+RANGE*j+RANGE*RANGE*k);
                    }
                }
            }
        }
        std::cout<<loopctr<<" "<<cur_err_max<<"\n";
        err_max = cur_err_max;
    }
    free(temp_field);
    free(temp);
    free(ddx);
    free(ddy);
    free(ddz);
    auto hst_en = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = hst_en-hst_st;
    std::cout<<"Duration: "<<duration.count()<<"\n";
    std::cout<<"With "<<loopctr<<" loops\n\n";
    std::cout<<"Error: "<<err_max<<"\n";
    device_list();
    //display(field);
    return 0;
}