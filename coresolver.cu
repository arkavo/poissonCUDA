#include <stdio.h>
//declare global limits
#define X_RANGE 64
#define Y_RANGE 64
#define Z_RANGE 64


int main()
{
    //fast setup
    float SPACE[X_RANGE][Y_RANGE][Z_RANGE];
    //initial mountain condition
    for(int i=0;i<X_RANGE;i++)
    {
        for(int j=0;j<Y_RANGE;j++)
        {
            for(int k=0;k<Z_RANGE;k++)
            {
                if((i+j+k>=30)&&(i+j+k<=34))
                    SPACE[i][j][k] = (i+j+k)/90;
                else
                    SPACE[i][j][k] = 0;
            }
        }        
    }
    
}