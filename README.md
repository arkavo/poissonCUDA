# poissonCUDA

This project is to discuss the advantages of using Parallel Programming to simulate the Poisson equation.
The first study will be to optimize a number of factors. We know that to solve the Poisson equation, we will have to use partial derivatives extensively. It is here, therefore that we turn our attention to. Optimizing for derivatives will give us more speed in the long run.

This experimental project will be a first step into creating an environment for a single GPU poisson solver. The 3D problem will be converted to a 2D mesh to facilitate parallelization since single GPUs in personal devices tend to have a limited number of cores. 

## Poisson Equation

The [Poisson Equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) is a partial differential equation with varied usage in physics. Its solution will vary with boundary conditions with varying periodicity. As a test case scenario, we have taken a statc boundary as our boundary condition but we are free to use any boundary conditions we choose.

## Mechanic

Given the boundary conditions, the quantities inside the volume will change until we reach a stable solution. Stability of a solution will be determined whether or not it changes below a threshold on its next iteration. We shall loop until the maximum error we find within is less than the tolerance alue of the solution.
