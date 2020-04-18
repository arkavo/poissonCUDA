#include <stdio.h>
//declare global limits
#define X_RANGE 256
#define Y_RANGE 256
//#define Z_RANGE 8

void data_divide(int dev_ct)
{

}

__global__ void d_x(int lx,int ly,float* ADR,float* RVE)
{
    int x = threadIdx.x*blockDim.x + blockIdx.x;
    int y = threadIdx.y*blockDim.y + blockIdx.y;
    ADR[x*blockDim.x + y] = RVE[x*blockDim.x + 1 + y] - RVE[x*blockDim.x + y];
}

int main()
{
    int dev_ct;
    cudaGetDeviceCount(&dev_ct);
    printf("%d\n",dev_ct);
    FILE *fp;
    fp = fopen("data.txt","w+");
    //fast setup
    float SPACE[X_RANGE][Y_RANGE];
    //  float TEMP[X_RANGE][Y_RANGE];
    //for dev_ct = 1
    float* ADD;
    ADD = (float*)malloc(X_RANGE*Y_RANGE*sizeof(float));
    float* TEMP;
    //TEMP = (float*)malloc(X_RANGE*Y_RANGE*sizeof(float));
    //  cudaMalloc(&TEMP,X_RANGE*Y_RANGE*sizeof(float));
    //initial mountain condition
    for(int i=0;i<X_RANGE;i++)
    {
        for(int j=0;j<Y_RANGE;j++)
        {
            SPACE[i][j] = i*(X_RANGE-i)-j*(Y_RANGE-j);
            //SPACE[i][j] = rand();
        }        
    }
    
    //cudaMemcpy(TEMP,SPACE,X_RANGE*Y_RANGE*sizeof(float),cudaMemcpyHostToDevice);
    
    for(int i=0;i<X_RANGE;i++)
    {
        for(int j=0;j<Y_RANGE;j++)
        {
            fprintf(fp,"%f ",SPACE[i][j]);
        }        
    }
    //cudaFree(TEMP);
}