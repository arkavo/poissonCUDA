#include <stdio.h>
#include <iostream>
#include <cmath>
#include <limits.h>

using namespace std;

#define X 32
#define Y 32
#define Z 32

__global__ void DX(double* R, double* C,double dx)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y +X+X*Y;
    for(int i=1;i<X-1;i++)
    {
        *(R+index+i) = (*(C+index+1+i) - *(C+index-1+i))/(dx*2);
    }
}

__global__ void DY(double* R, double* C,double dy)
{
    int index = threadIdx.x + blockIdx.x*X*Y+ X + X*Y;
    for(int i=1;i<Y-1;i++)
    {
        *(R+index+i) = (*(C+index+X*(i+1)) - *(C+index+X*(i-1)))/(dy*2);
    }
}

__global__ void DZ(double* R, double* C,double dz)
{
    int index = threadIdx.x + blockIdx.x*X+X+X*Y;
    for(int i=1;i<Z-1;i++)
    {
        *(R+index+i) = (*(C+index+X*Y*(i+1))-*(C+index+X*Y*(i-1)))/(dz*2);
    }
}

__global__ void DDX(double* R, double* C,double dx)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y + X+X*Y;
    for(int i=1;i<X-1;i++)
    {
        *(R+index+i) = (*(C+index+1+i) + *(C+index-1+i) - 2* *(C+index+i))/(dx*dx);
    }
}

__global__ void DDY(double* R, double* C,double dy)
{
    int index = threadIdx.x + blockIdx.x*X*Y+X+X*Y;
    for(int i=1;i<Y-1;i++)
    {
        *(R+index+i) = (*(C+index+X*(i+1)) + *(C+index+X*(i-1)) - 2* *(C+index+i))/(dy*dy);
    }
}

__global__ void DDZ(double* R, double* C,double dz)
{
    int index = threadIdx.x + blockIdx.x*X+X+X*Y;
    for(int i=0;i<Z-1;i++)
    {
        *(R+index+i) = (*(C+index+X*Y*(i+1))+*(C+index+X*Y*(i-1))-2* *(C+index+i))/(dz*dz);
    }
}

__global__ void ASSIGN(double* R, double* C)
{
    int index = threadIdx.x*X + blockIdx.x*X*Y + X + X*Y ;
    for(int i=1;i<X-1;i++)
    {
        *(R+index+i) = *(C+index+i);
    }
}

__global__ void ADD(double* R,double* C,double dt)
{
    int index = threadIdx.x*X+blockIdx.x*X*Y+X+X*Y;
    for(int i=0;i<X;i++)
    {
        *(R+index+i) += (*(C+index+i) * dt);
    }
}

__global__ void COMPARE(double* R, double* C, double* OUT_H)
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
__global__ void RESET_CTR(double* C)
{
    *C = 0;
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
                        DATA_H[i][j][k] = i+j+k;
                    else
                        DATA_H[i][j][k] = 0;
    }}}
    
    double step = 0.01;
    double* CC;
    double CCD = 0;

    double tol = 0.00001;

    double* DATA_ORIGINAL;
    //double* DATA_TRIMMED;
    double* DATA_NEXT;
    double* DX_D;
    double* DY_D;
    double* DZ_D;
    double* DDX_D;
    double* DDY_D;
    double* DDZ_D;
    
    unsigned long long SIZE_0 = ((int)sizeof(double))*X*Y*Z;
    //unsigned long long SIZE_1 = sizeof(double)*(X-2)*(Y-2)*(Z-2);
    
    cudaMalloc((void**)&DATA_ORIGINAL,SIZE_0);
    //cudaMalloc(&DATA_TRIMMED,SIZE_1);
    cudaMalloc((void**)&DATA_NEXT,SIZE_0);
    
    cudaMalloc((void**)&DX_D,SIZE_0);
    cudaMalloc((void**)&DY_D,SIZE_0);
    cudaMalloc((void**)&DZ_D,SIZE_0);
    cudaMalloc((void**)&DDX_D,SIZE_0);
    cudaMalloc((void**)&DDY_D,SIZE_0);
    cudaMalloc((void**)&DDZ_D,SIZE_0);
    
    cudaMalloc(&CC,(int)sizeof(double));
    //cudaMemcpy(CC,&CCD,sizeof(double),cudaMemcpyHostToDevice);
    CCD = 10;

    cudaMemcpy(DATA_ORIGINAL,&DATA_H,SIZE_0,cudaMemcpyHostToDevice);
    cudaMemcpy(DATA_NEXT,&DATA_H,SIZE_0,cudaMemcpyHostToDevice);
    double DATA_F[X][Y][Z];
    double DATA_FF[X][Y][Z];
    
    cudaMemcpy(&DATA_FF,DATA_ORIGINAL,SIZE_0,cudaMemcpyDeviceToHost);
    //display(&DATA_FF[0][0][0]);
    //ASSIGN<<<Y,Z>>>(DATA_TRIMMED,DATA_ORIGINAL);
    int ct = 0;
    //ADD <<<Y,Z>>> (DATA_NEXT,DATA_ORIGINAL,1);
    while(CCD>=tol)
    {
        RESET_CTR  <<<1,1>>>  (CC);
        //printf("%lf\r",*CC);
        //DX <<<Y-2,Z-2>>> (DX_D,DATA_ORIGINAL,1.);
        //DY <<<Z-2,X-2>>> (DY_D,DATA_ORIGINAL,1.);
        //DZ <<<Z-2,X-2>>> (DZ_D,DATA_ORIGINAL,1.);

        DDY <<<Z-2,X-2>>> (DDY_D,DATA_ORIGINAL,10.);
        DDZ <<<X-2,Y-2>>> (DDZ_D,DATA_ORIGINAL,10.);
        DDX <<<Y-2,Z-2>>> (DDX_D,DATA_ORIGINAL,10.);
        
        //ADD <<<Y-2,Z-2>>> (DATA_NEXT,DX_D,-1*step);        
        //ADD <<<Y-2,Z-2>>> (DATA_NEXT,DY_D,-1*step);
        //ADD <<<Y-2,Z-2>>> (DATA_NEXT,DZ_D,-1*step);
        ADD <<<Y-2,Z-2>>> (DATA_NEXT,DDX_D,step);
        ADD <<<Y-2,Z-2>>> (DATA_NEXT,DDY_D,step);
        ADD <<<Y-2,Z-2>>> (DATA_NEXT,DDZ_D,step);
        
        COMPARE<<<Y-2,Z-2>>>(DATA_ORIGINAL,DATA_NEXT,CC);
        
        cudaMemcpy(&CCD,CC,sizeof(double),cudaMemcpyDeviceToHost);
        //printf("%0.3lf\r",CCD);
        
        ASSIGN  <<<Y-2,Z-2>>>  (DATA_ORIGINAL,DATA_NEXT);
        
        ct += 1;
        if(ct%1000==0)
        {
            cudaMemcpy(&CCD,CC,sizeof(double),cudaMemcpyDeviceToHost);
            printf("%d loops %0.6lf max error\r",ct,CCD);
        }
        //*CC = 0;
    }
    //double DATA_F[X][Y][Z];
    cudaMemcpy(DATA_F,DATA_ORIGINAL,SIZE_0,cudaMemcpyDeviceToHost);
    printf("\nConverged in %d loops\n",ct-1);
    printf("%d\n",SIZE_0);
    //display(&DATA_F[0][0][0]);
    cudaFree(DATA_ORIGINAL);
    //cudaFree(DATA_TRIMMED);
    cudaFree(DATA_NEXT);
    cudaFree(DX_D);
    cudaFree(DY_D);
    cudaFree(DZ_D);
    cudaFree(DDX_D);
    cudaFree(DDY_D);
    cudaFree(DDZ_D);
    
    return 0;
}