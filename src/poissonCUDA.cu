//Three Dimensional Poisson solver
//Author: Arkavo Hait, 2021


#include <stdio.h>
#include <iostream>
#include <cmath>
#include <limits.h>

using namespace std;

//Define Global variables

//#define X 5
//#define Y 5
//#define Z 5

//Box struct
struct BOX
{
    int X;
    int Y;
    int Z;
};
/*
//X derivative (unused)

__global__ void DX(double* R, double* C,double dx)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y +X+X*Y;
    for(int i=1;i<X-1;i++)
    {
        *(R+index+i) = (*(C+index+1+i) - *(C+index-1+i))/(dx*2);
    }
}
//Y derivative (unused)
__global__ void DY(double* R, double* C,double dy)
{
    int index = threadIdx.x + blockIdx.x*X*Y+ X + X*Y;
    for(int i=1;i<Y-1;i++)
    {
        *(R+index+i) = (*(C+index+X*(i+1)) - *(C+index+X*(i-1)))/(dy*2);
    }
}
//Z derivative (unused)
__global__ void DZ(double* R, double* C,double dz)
{
    int index = threadIdx.x + blockIdx.x*X+X+X*Y;
    for(int i=1;i<Z-1;i++)
    {
        *(R+index+i) = (*(C+index+X*Y*(i+1))-*(C+index+X*Y*(i-1)))/(dz*2);
    }
}
*/
//X double derivative
__global__ void DDX(double* R, double* C,int X,int Y,int Z,double dx)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y + X+X*Y;
    for(int i=1;i<X-1;i++)
    {
        *(R+index+i) = (*(C+index+1+i) + *(C+index-1+i) - 2* *(C+index+i))/(dx*dx);
    }
}

//Y double derivative
__global__ void DDY(double* R, double* C,int X,int Y,int Z, double dy)
{
    int index = threadIdx.x + blockIdx.x*X*Y+X+X*Y;
    for(int i=1;i<Y-1;i++)
    {
        *(R+index+i*X) = (*(C+index+X*(i+1)) + *(C+index+X*(i-1)) - 2* *(C+index+i*X))/(dy*dy);
    }
}

//Z double derivative
__global__ void DDZ(double* R, double* C,int X,int Y,int Z, double dz)
{
    int index = threadIdx.x + blockIdx.x*X+X+X*Y;
    for(int i=0;i<Z-1;i++)
    {
        *(R+index+i*X*Y) = (*(C+index+X*Y*(i+1)) + *(C+index+X*Y*(i-1))- 2* *(C+index+i*X*Y))/(dz*dz);
    }
}

//parallel function to update matrices
__global__ void ASSIGN(double* R, double* C,int X,int Y,int Z)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y + X + X*Y ;
    for(int i=1;i<X-1;i++)
    {
        *(R+index+i) = *(C+index+i);
    }
}

//parallel function to add two matrices
__global__ void ADD(double* R,double* C,double dt,int X,int Y,int Z)
{
    int index = threadIdx.x*X+blockIdx.x*X*Y+X+X*Y;
    for(int i=0;i<X;i++)
    {
        *(R+index+i) += (*(C+index+i) * dt);
    }
}

//parallel function to compare two matrices, outputting a maximum difference bteween elements
__global__ void COMPARE(double* R, double* C, double* OUT_H,int X,int Y,int Z)
{
    int index = threadIdx.x * X + blockIdx.x * X * Y+X+X*Y;
    for(int i=1;i<X-1;i++)
    {
        if(abs(*(R+index+i)-*(C+index+i))>= *OUT_H)
        {
            *OUT_H = abs(*(R+index+i) - *(C+index+i));
        }
    }
}

//reset function, use it to reset any pointer
__global__ void RESET_CTR(double* C)
{
    *C = 0;
}

//display function, use to disaply any array
void display(double* DATA,int X,int Y,int Z)
{
    for(int k=0;k<Z;k++)
        {for(int j=0;j<Y;j++)
            {for(int i=0;i<X;i++)
                {printf("%.2lf ",*(DATA+i+X*j+X*Z*k));}
            printf("\n");}printf("\n");}
}

//main fxn, will fix with args after
int main()
{
    //DECLARE YOUR VARIABLES HERE
    
    struct BOX grid = {30,30,30};
    //step for double derivatives
    double step = 0.01;
    //tolerence
    double tol = 0.00001;
    
    const int X = grid.X;
    const int Y = grid.Y;
    const int Z = grid.Z;
    printf("\n%d %d %d\n",X,Y,Z);
    unsigned long long SIZE_0 = ((int)sizeof(double))*X*Y*Z;
    double* DATA_H; 
    double* DATA_F;
    DATA_H = (double*)malloc(SIZE_0);
    DATA_F = (double*)malloc(SIZE_0);
    
    //Impose Boundary conditions here
    
    for(int i=0;i<X;i++)
        {for(int j=0;j<Y;j++)
            {for(int k=0;k<Z;k++)
                {
                    if(i==0||i==(X-1)||j==0||j==(Y-1)||k==0||k==(Z-1))
                        *(DATA_H+i+j*X+k*X*Y) = 5;
                    else
                        *(DATA_H+i+j*X+k*X*Y) = 0;
    }}}
    
    
    //error counters
    double* CC;
    double CCD = 0;
    
    //DATA pointer for device state 0
    double* DATA_ORIGINAL;
    //DATA pointer for device state 1
    double* DATA_NEXT;
    //Derivative results pointers
    double* DDX_D;
    double* DDY_D;
    double* DDZ_D;
    
    //array size for device
    
    //allocating space for arrays
    cudaMalloc((void**)&DATA_ORIGINAL,SIZE_0);
    
    cudaMalloc((void**)&DATA_NEXT,SIZE_0);
    
    cudaMalloc((void**)&DDX_D,SIZE_0);
    cudaMalloc((void**)&DDY_D,SIZE_0);
    cudaMalloc((void**)&DDZ_D,SIZE_0);
    
    cudaMalloc(&CC,(int)sizeof(double));
    
    //token counter
    CCD = 10;
    //copy data state0, state1 
    cudaMemcpy(DATA_ORIGINAL,DATA_H,SIZE_0,cudaMemcpyHostToDevice);
    cudaMemcpy(DATA_NEXT,DATA_H,SIZE_0,cudaMemcpyHostToDevice);
    //data arrays for final copies
    //double DATA_F[X][Y][Z];
    
    //double DATA_FF[X][Y][Z];
    
    //cudaMemcpy(&DATA_FF,DATA_ORIGINAL,SIZE_0,cudaMemcpyDeviceToHost);
    
    //counter
    int ct = 0;
    
    //run while tolerence > difference
    while(CCD>tol)
    {
        //reset difference every loop
        RESET_CTR  <<<1,1>>>  (CC);
        //run derivatives
        DDY <<<Z-2,X-2>>> (DDY_D,DATA_ORIGINAL,X,Y,Z,10.);
        DDZ <<<X-2,Y-2>>> (DDZ_D,DATA_ORIGINAL,X,Y,Z,10.);
        DDX <<<Y-2,Z-2>>> (DDX_D,DATA_ORIGINAL,X,Y,Z,10.);     //add into state 1
        ADD <<<Y-2,Z-2>>> (DATA_NEXT,DDX_D,step,X,Y,Z);
        ADD <<<Y-2,Z-2>>> (DATA_NEXT,DDY_D,step,X,Y,Z);
        ADD <<<Y-2,Z-2>>> (DATA_NEXT,DDZ_D,step,X,Y,Z);
        //compare state1 state 0
        COMPARE<<<Y-2,Z-2>>>(DATA_ORIGINAL,DATA_NEXT,CC,X,Y,Z);
        //copy back max error
        cudaMemcpy(&CCD,CC,sizeof(double),cudaMemcpyDeviceToHost);
        //make state 1 as state0
        ASSIGN  <<<Y-2,Z-2>>>  (DATA_ORIGINAL,DATA_NEXT,X,Y,Z);
        //update counter
        ct += 1;
        //information every 1000 loops because of visibility
        if(ct%1000==0)
        {
            cudaMemcpy(&CCD,CC,sizeof(double),cudaMemcpyDeviceToHost);
            printf("%d loops %0.6lf max error\r",ct,CCD);
        }
    }
    //copy back final array
    cudaMemcpy(DATA_F,DATA_ORIGINAL,SIZE_0,cudaMemcpyDeviceToHost);
    //final print statement
    printf("\n\nConverged in %d loops\n\n",ct-1);
    //display optional
    //display(DATA_F,X,Y,Z);
    //free pointers
    cudaFree(DATA_ORIGINAL);
    cudaFree(DATA_NEXT);
    cudaFree(DDX_D);
    cudaFree(DDY_D);
    cudaFree(DDZ_D);
    
    return 0;
}