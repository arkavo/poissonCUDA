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
#define RANGE 512
#define ITER 10000
#define tol 0.01

float field[RANGE][RANGE][RANGE];
float temp_field[RANGE][RANGE][RANGE];

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
                    field[i][j][k] = 2;
                else
                    field[i][j][k] = 0;
            }
        }
    }
    //display();
    
    //Linear cpu time
    float err_max = 10.0;
    int loopctr = 0;
    auto hst_st = std::chrono::high_resolution_clock::now();
    while(err_max>tol)
    {
        for(int i=1;i<RANGE-1;i++)
        {
            for(int j=1;j<RANGE-1;j++)
            {
                for(int k=1;k<RANGE-1;k++)
                {
                    temp_field[i][j][k] = (field[i-1][j][k]+field[i+1][j][k]
                                        +field[i][j-1][k]+field[i][j+1][k]
                                        +field[i][j][k-1]+field[i][j][k+1])/6.0;      
                }
            }
        }
        for(int i=1;i<RANGE-1;i++)
        {
            for(int j=1;j<RANGE-1;j++)
            {
                for(int k=1;k<RANGE-1;k++)
                {
                    err_max = temp_field[i][j][k] - field[i][j][k];
                    loopctr++;
                    //if(loopctr%2000000==0)
                    //    std::cout<<loopctr<<" iters\n";
                    field[i][j][k] = temp_field[i][j][k];
                }
            }
        }
        
    }
    auto hst_en = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = hst_en-hst_st;
    printf("Duration : %f\n", duration.count());
    std::cout<<"With "<<loopctr<<" loops\n\n";
    //display();
}