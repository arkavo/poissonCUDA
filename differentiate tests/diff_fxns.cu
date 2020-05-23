#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <chrono>
#include <string.h>
#include <fstream>

#define RANGE 2048
#define ITER 10000

__global__ void diff(float* dat,float* drv,int order)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    drv[x+y*blockDim.x] = dat[x+y*blockDim.x+1] - dat[x+y*blockDim.x];
}


void device_diff(float* dat,float* dev)
{
    float* st_dat = dat;
    float* st_dev = dev;
    *dev = *(st_dat+1) - *dat;
    for(int i=0;i<RANGE-1;i++)
    {
        *(st_dev+i+1) = *(st_dat+1+i) - *(st_dat+i);
    }
}
int main(int argc,char* argv[])
{
    int order = (int)strtol(argv[1],NULL,10);
    //printf("arg = %d\n",order);
    FILE* fp;
    std::ofstream exc ("data_sheet.csv", std::ofstream::out);
    //exc.open("data_sheet.csv");
    exc << "Order, CPU_t, GPU_in_t, GPU_compute_t, GPU_ot_t\n";

    fp = fopen("linear_diff.txt","w+");
    float data_host[RANGE];
    float derv_host[RANGE];
    float res[RANGE];
    float* data_dev;
    float* derv_dev;
    //timer setup
    srand(time(NULL));
    cudaMalloc(&data_dev,sizeof(float)*RANGE);
    cudaMalloc(&derv_dev,sizeof(float)*RANGE);
    for(int i = 0;i<RANGE;i++)
    {
        data_host[i] = rand()%100;
        //printf("%f ",data_host[i]);
    }
    
    auto hst_st = std::chrono::high_resolution_clock::now();
    for(int i=0;i<ITER;i++)
        device_diff(data_host,derv_host);
    auto hst_en = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float> duration = hst_en-hst_st;
    
    printf("Duration : %f\n", duration.count());
    
    auto dev_cp_st = std::chrono::high_resolution_clock::now();
    cudaMemcpy(data_dev,data_host,sizeof(float)*RANGE,cudaMemcpyHostToDevice);
    auto dev_cp_en = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float> copy_dev = dev_cp_en - dev_cp_st;

    auto dev_op_st = std::chrono::high_resolution_clock::now();
    for(int i=0;i<ITER;i++)
        diff<<< (int)(2000/order) , order>>>(data_dev,derv_dev,order);
    auto dev_op_en = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float> dev_op = dev_op_en - dev_op_st;

    auto hst_cp_st = std::chrono::high_resolution_clock::now();
    cudaMemcpy(res,derv_dev,sizeof(float)*RANGE,cudaMemcpyDeviceToHost);
    auto hst_cp_en = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float> hst_cp = hst_cp_en - hst_cp_st;

    printf("Duration of GPU = %f cpy + %f op + %f res\n",copy_dev.count(),dev_op.count(),hst_cp.count());
    printf("Order = %d\n",order);
    for(int i=0;i<RANGE;i++)
    {     
        //printf("%.2f ",derv_host[i]);
        fprintf(fp,"%.2f ",derv_host[i]);
    }
    exc <<order
        <<(float)duration.count()
        <<(float)copy_dev.count()
        <<(float)dev_op.count()
        <<(float)hst_cp.count();
    exc.close();
    return 0;
}