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
#define RANGE 256
#define ITER 10000
#define tol 0.001
#define dr 1
#define dt 0.2

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
    //display();
    
    //Linear cpu time
    float err_max = 10.0;
    int loopctr = 0;
    auto hst_st = std::chrono::high_resolution_clock::now();
    while(abs(err_max)>tol)
    {
        deriv(1);
        deriv(2);

        
        for(int i=0;i<RANGE;i++)
        {
            for(int j=0;j<RANGE;j++)
            {
                for(int k=0;k<RANGE;k++)
                {
                    dx[i][j][k]+=(ddx[i][j][k])*dt;
                    dy[i][j][k]+=(ddy[i][j][k])*dt;
                    dz[i][j][k]+=(ddz[i][j][k])*dt;
                }
            }
        }
        for(int i=1;i<RANGE-1;i++)
        {
            for(int j=1;j<RANGE-1;j++)
            {
                for(int k=1;k<RANGE-1;k++)
                {
                    temp_field[i][j][k] = field[i][j][k]+(dx[i][j][k]+dy[i][j][k]+dz[i][j][k])*dt;      
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
    std::cout<<"Duration: "<<duration.count()<<"\n";
    std::cout<<"With "<<loopctr<<" loops\n\n";
    std::cout<<"Error: "<<err_max<<"\n";
    //display();
}