# poissonCUDA

This project is to discuss the advantages of using Parallel Programming to simulate the Poisson equation.
The first study will be to optimize a number of factors. We know that to solve the Poisson equation, we will have to use partial derivatives extensively. It is here, therefore that we turn our attention to. Optimizing for derivatives will give us more speed in the long run.

This experimental project will be a first step into creating an environment for a single GPU poisson solver. The 3D problem will be converted to a 2D mesh to facilitate parallelization since single GPUs in personal devices tend to have a limited number of cores. 

## Poisson Equation

The [Poisson Equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) is a partial differential equation with varied usage in physics. Its solution will vary with boundary conditions with varying periodicity. As a test case scenario, we have taken a statc boundary as our boundary condition but we are free to use any boundary conditions we choose.

## Algorithm

Given the boundary conditions, the quantities inside the volume will adjust, starting from 0 until we reach a stable solution. Stability of a solution will be determined whether or not it changes below a threshold on its next iteration. We shall loop until the maximum error we find within is less than the tolerance value of the solution.

## Mechanic

We will feed the XYZ matrix into the GPU. Due to boundary conditions being strictly on the walls, we can use the matrix for computation as (X-2)(Y-2)(Z-2). We will define an index based on the new matrix in each of our functions and then will map it to our **Block, Thread** system where the total number of blocks x threads will be equal to or just more than (X-2)(Y-2)(Z-2) to facilitate threading. This is to ensure that we can directly specify the number of *threads* in our execution.

## Results

We will be comparing converge times with respect to data size and threads. Since this is fully done on a local machine right now. Please excuse the low data capacities. I will be making a study on large data as soon as I have access to a sufficiently powerful machine.
