#include <stdio.h>

#define X 5
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

__global__ void DXX(double* R, double* C,double dx)
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
    
    double* datap = &DATA_H[0][0][0];
    display(datap);
    
    return 0;
}