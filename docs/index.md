# poissonCUDA

This project is to discuss the advantages of using Parallel Programming to simulate the Poisson equation.
The first study will be to optimize a number of factors. We know that to solve the Poisson equation, we will have to use partial derivatives extensively. It is here, therefore that we turn our attention to. Optimizing for derivatives will give us more speed in the long run.

This experimental project will be a first step into creating an environment for a single GPU poisson solver. The 3D problem will be converted to a 2D mesh to facilitate parallelization since single GPUs in personal devices tend to have a limited number of cores. 

## Poisson Equation

The [Poisson Equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) is a partial differential equation with varied usage in physics. Its solution will vary with boundary conditions with varying periodicity. As a test case scenario, we have taken a statc boundary as our boundary condition but we are free to use any boundary conditions we choose.

## Algorithm

Given the boundary conditions, the quantities inside the volume will adjust, starting from 0 until we reach a stable solution. Stability of a solution will be determined whether or not it changes below a threshold on its next iteration. We shall loop until the maximum error we find within is less than the tolerance value of the solution.

## Mechanic

We will feed the XYZ matrix into the GPU. Due to boundary conditions being strictly on the walls, we can use the matrix for computation as (X-2)(Y-2)(Z-2). We will define an index based on the new matrix in each of our functions and then will map it to our **Block, Thread** system where the total number of blocks x threads will be equal to or just more than (X-2)(Y-2)(Z-2) to facilitate threading. This is to ensure that we can directly specify the number of *threads* in our execution. On a side note, please be aware that I have used serial code for X and parallel code for Y and Z dimensions. This means that there is potentially 1 *more* level of parallelization to be explored which I will do at a later date. 

## Usage

The main source file is under src folder. You may need to recomplie it as you see fit. Preferably run **nvcc** on it. You can then create an executable for your needs. There are 2 options for command line arguments as **pCUDA Threads X Y Z**. X, Y, Z are optional parameters which default to 10 upon not entering. Threads is a compulsory parameter which is the number of threads you will need for computation. If unsure try running with **Threads = 1**.

## Results

We will be comparing converge times with respect to data size and threads. Since this is fully done on a local machine right now. Please excuse the low data capacities. I will be making a study on large data as soon as I have access to a sufficiently powerful machine.

Threads|X|Y|Z|Size|Execution Time(s)
----|------|------|------|---------|---------
1   |  10  |  10  |  10  |  1000   |  8.12449
1   |  20  |  20  |  20  |  8000   |  29.7687
1   |  30  |  30  |  30  |  27000  |  151.321
2   |  10  |  10  |  10  |  1000   |  7.20126
2   |  20  |  20  |  20  |  8000   |  13.307
2   |  30  |  30  |  30  |  27000  |  39.3812
4   |  10  |  10  |  10  |  1000   |  7.01841
4   |  20  |  20  |  20  |  8000   |  10.3198
4   |  30  |  30  |  30  |  27000  |  20.911
8   |  10  |  10  |  10  |  1000   |  7.10812
8   |  20  |  20  |  20  |  8000   |  9.63892
8   |  30  |  30  |  30  |  27000  |  14.8858
16  |  10  |  10  |  10  |  1000   |  7.16111
16  |  20  |  20  |  20  |  8000   |  9.42222
16  |  30  |  30  |  30  |  27000  |  12.8313

![fig](https://user-images.githubusercontent.com/16555024/119909307-2c93e900-bf72-11eb-8079-c8134f6589c2.png)

We can see a drastic change in computation times in the GPU as the aggregate data size goes up. From 150s in single thread to a mere 39s in double threads, which almost a *4x* speedup. Going further till 4, 8 and 16 threads per block we see faster results as *7.5x, 10x, 12x* speedups on average. I was surprised to see a speedup in 16 cores since I believed my own machine to cap out at 8 threads/block. If this scale gets too improper to understand, we can try seeing this on a log scale as:

![log_fig](https://user-images.githubusercontent.com/16555024/119910066-ec356a80-bf73-11eb-8c82-833529214fae.png).

This figure exists to inform us about the extensive speedups we get when the order of magnitude of data points change.

## Personal Thoughts

I will probably shelve this project for now since I don't own a more powerful machine. As a roadmap I'll try to create a header file for all the functions I've used till now. Anyone is elcome to contact me for more requests or any other questions they might have for this mini project.

**-Line**
