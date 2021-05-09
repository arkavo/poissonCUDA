#include <stdio.h>
#include <iostream>
#include <cmath>

#define X 4
#define Y 4
#define Z 4

__global__ void DX(double* R, double* C,double dx)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y;
    for(int i=1;i<X-1;i++)
    {
        R[index] = (C[index+1]-C[index-1])/(dx*2);
    }
}

__global__ void DY(double* R, double* C,double dy)
{
    int index = threadIdx.x + blockIdx.x*X*Y;
    for(int i=1;i<Y-1;i++)
    {
        R[index] = (C[index+X]-C[index-X])/(dy*2);
    }
}

__global__ void DZ(double* R, double* C,double dz)
{
    int index = threadIdx.x + blockIdx.x*X;
    for(int i=0;i<Z-1;i++)
    {
        R[index] = (C[index+X*Y]-C[index-X*Y])/(dz*2);
    }
}

__global__ void DDX(double* R, double* C,double dx)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y;
    R[index] = (C[index+1]+C[index-1]-2*C[index])/(dx*dx);
}

__global__ void DDY(double* R, double* C,double dy)
{
    int index = threadIdx.x + blockIdx.x*X*Y;
    for(int i=1;i<Y-1;i++)
    {
        R[index] = (C[index+X]+C[index-X]-2*C[index])/(dy*dy);
    }
}

__global__ void DDZ(double* R, double* C,double dz)
{
    int index = threadIdx.x + blockIdx.x*X;
    for(int i=0;i<Z-1;i++)
    {
        R[index] = (C[index+X*Y]+C[index-X*Y]-2*C[index])/(dz*dz);
    }
}

__global__ void ASSIGN(double* R, double* C)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y;
    for(int i=1;i<X-1;i++)
    {
        R[index+i-1+X*Y+X] = C[index+i+X*Y+X];
    }
}

__global__ void ADD(double* R,double* C,double dt)
{
    int index = threadIdx.x*X+blockIdx.x*X*Y;
    for(int i=0;i<X;i++)
    {
        R[index+i] += C[index+i] * dt;
    }
}

__global__ void COMPARE(double* R, double*C, double* OUT_H)
{
    int index = threadIdx.x*X+blockIdx.x*X*Y;
    for(int i=0;i<X;i++)
    {
        if(abs(R[index]-C[index])>=*OUT_H)
        {
            *OUT_H = abs(R[index]-C[index]);
        }
    }
}
void display(double* DATA)
{
    for(int i=0;i<X;i++)
        {for(int j=0;j<Y;j++)
            {for(int k=0;k<Z;k++)
                {printf("%.2lf ",*(DATA+i+X*j+X*Z*k));}
            printf("\n");}printf("\n");}
}
int main()
{
    double DATA_H[X][Y][Z];
    for(int i=0;i<X;i++)
        {for(int j=0;j<Y;j++)
            {for(int k=0;k<Z;k++)
                {
                    if(i==0||i==(X-1)||j==0||j==(Y-1)||k==0||k==(Z-1))
                        DATA_H[i][j][k] = 2;
                    else
                        DATA_H[i][j][k] = 0;
    }}}
    
    double step = 0.01;
    double* CC;
    double CCD = 0;

    double tol = 0.001;

    double* DATA_ORIGINAL;
    //double* DATA_TRIMMED;
    double* DATA_NEXT;
    double* DX_D;
    double* DY_D;
    double* DZ_D;
    double* DDX_D;
    double* DDY_D;
    double* DDZ_D;
    
    unsigned long long SIZE_0 = sizeof(double)*X*Y*Z;
    //unsigned long long SIZE_1 = sizeof(double)*(X-2)*(Y-2)*(Z-2);
    
    cudaMalloc(&DATA_ORIGINAL,SIZE_0);
    //cudaMalloc(&DATA_TRIMMED,SIZE_1);
    cudaMalloc(&DATA_NEXT,SIZE_0);
    
    cudaMalloc(&DX_D,SIZE_0);
    cudaMalloc(&DY_D,SIZE_0);
    cudaMalloc(&DZ_D,SIZE_0);
    cudaMalloc(&DDX_D,SIZE_0);
    cudaMalloc(&DDY_D,SIZE_0);
    cudaMalloc(&DDZ_D,SIZE_0);
    
    cudaMalloc(&CC,sizeof(double));
    //cudaMemcpy(CC,&CCD,sizeof(double),cudaMemcpyHostToDevice);
    *CC = 0;
    CCD = 1;

    cudaMemcpy(DATA_ORIGINAL,DATA_H,SIZE_0,cudaMemcpyHostToDevice);
    //ASSIGN<<<Y,Z>>>(DATA_TRIMMED,DATA_ORIGINAL);
    int ct = 0;
    
    while(CCD>=tol)
    {
        DX<<<Y,Z>>>(DX_D,DATA_ORIGINAL,10);
        DY<<<Z,X>>>(DY_D,DATA_ORIGINAL,10);
        DZ<<<Z,X>>>(DZ_D,DATA_ORIGINAL,10);

        DDX<<<Y,Z>>>(DDX_D,DATA_ORIGINAL,10);
        DDY<<<Z,X>>>(DDY_D,DATA_ORIGINAL,10);
        DDZ<<<X,Y>>>(DDZ_D,DATA_ORIGINAL,10);
        
        ADD<<<Y,Z>>>(DATA_NEXT,DX_D,step);
        ADD<<<Y,Z>>>(DATA_NEXT,DY_D,step);
        ADD<<<Y,Z>>>(DATA_NEXT,DZ_D,step);
        ADD<<<Y,Z>>>(DATA_NEXT,DDX_D,step*step);
        ADD<<<Y,Z>>>(DATA_NEXT,DDY_D,step*step);
        ADD<<<Y,Z>>>(DATA_NEXT,DDZ_D,step*step);
        COMPARE<<<Y,Z>>>(DATA_NEXT,DATA_ORIGINAL,CC);
        cudaMemcpy(&CCD,CC,sizeof(double),cudaMemcpyDeviceToHost);
        //printf("%0.3lf\r",CCD);
        ASSIGN<<<Y-2,Z-2>>>(DATA_ORIGINAL,DATA_NEXT);
        ct += 1;
        if(ct%1000==0)
        {
            printf("%d loops %0.3lf max error\r",ct,CCD);
        }
        *CC = 0;
    }
    printf("Converged in %d loops\n",ct);
    cudaFree(DATA_ORIGINAL);
    //cudaFree(DATA_TRIMMED);
    cudaFree(DATA_NEXT);
    cudaFree(DX_D);
    cudaFree(DY_D);
    cudaFree(DZ_D);
    cudaFree(DDX_D);
    cudaFree(DDY_D);
    cudaFree(DDZ_D);
    //display(datap);
    return 0;
}