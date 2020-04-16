#include <stdio.h>
//declare global limits
#define X_RANGE 8
#define Y_RANGE 8
//#define Z_RANGE 8


int main()
{
    FILE *fp;
    fp = fopen("data.txt","w+");
    //fast setup
    float SPACE[X_RANGE][Y_RANGE];
    //initial mountain condition
    for(int i=0;i<X_RANGE;i++)
    {
        for(int j=0;j<Y_RANGE;j++)
        {
            if(i>=3&&j>=3&&i<=5&&j<=5)
                SPACE[i][j] = (i+j)/10.0;
            else
                SPACE[i][j] = 0;
        }        
    }
    
    for(int i=0;i<X_RANGE;i++)
    {
        for(int j=0;j<Y_RANGE;j++)
        {
            fprintf(fp,"%f ",SPACE[i][j]);
        }        
    }
}