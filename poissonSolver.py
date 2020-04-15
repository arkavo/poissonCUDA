import numpy as np

# Limits along each direction
# L - Along X
# M - Along Y
# N - Along Z

N, M, L = 64, 64, 64

# Tolerance value in Jacobi iterations
tolerance = 0.00001

# Main segment of code.
def main():
    global N, M, L

    # There are extra points in these fields (LHS and RHS) as this code is taken from a CFD solver.
    # The solver uses ghost points to impose boundary conditions.
    # The boundary points are 0 and -1
    # Jacobi iterations take place from 1 to -2

    lhs = np.ones([N+1, M+1, L+1])
    rhs = np.ones([N+1, M+1, L+1])

    # Define grid spacings
    hx = 1.0/(L-1)
    hy = 1.0/(M-1)
    hz = 1.0/(N-1)

    # NOTE
    # The Poisson solver will NOT converge as RHS is 1.0 everywhere
    # Please use the required function as RHS before calling Poisson solver
    lhs = poisson(rhs, hx, hy, hz)


#Poisson solver
def poisson(H, hx, hy, hz):
    global N, M, L

    P = solve(H, hx, hy, hz)
    chMat = laplace(P, hx, hy, hz)

    erMat = np.abs(H[1:N, 1:M, 1:L] - chMat[1:N, 1:M, 1:L])
    print "Error after poisson solver is ", np.amax(erMat)

    return P


#This function uses the Jacobi iterative solver, using the grid spacing
def solve(rho, hx, hy, hz):
    global tolerance

    # 1 subtracted from shape to account for ghost points
    [N, M, L] = np.array(np.shape(rho)) - 1
    prev_sol = np.zeros_like(rho)
    next_sol = np.zeros_like(rho)
    jCnt = 0

    while True:
        next_sol[1:N, 1:M, 1:L] = (
            (hy*hy)*(hz*hz)*(prev_sol[1:N, 1:M, 2:L+1] + prev_sol[1:N, 1:M, 0:L-1]) +
            (hx*hx)*(hz*hz)*(prev_sol[1:N, 2:M+1, 1:L] + prev_sol[1:N, 0:M-1, 1:L]) +
            (hx*hx)*(hy*hy)*(prev_sol[2:N+1, 1:M, 1:L] + prev_sol[0:N-1, 1:M, 1:L]) -
            (hx*hx)*(hy*hy)*(hz*hz)*rho[1:N, 1:M, 1:L])/ \
      (2.0*((hy*hy)*(hz*hz) + (hx*hx)*(hz*hz) + (hx*hx)*(hy*hy)))

        solLap = np.zeros_like(next_sol)
        solLap[1:N, 1:M, 1:L] = DDXi(next_sol, N, M, L, hx) + DDEt(next_sol, N, M, L, hy) + DDZt(next_sol, N, M, L, hz)

        error_temp = np.abs(rho[1:N, 1:M, 1:L] - solLap[1:N, 1:M, 1:L])
        maxErr = np.amax(error_temp)
        print maxErr
        if maxErr < tolerance:
            break

        jCnt += 1
        if jCnt > 10*N*M*L:
            print "ERROR: Jacobi not converging. Aborting"
            print "Maximum error: ", maxErr
            quit()

        prev_sol = np.copy(next_sol)

    return prev_sol


def laplace(function, hx, hy, hz):
    # 1 subtracted from shape to account for ghost points
    [N, M, L] = np.array(np.shape(function)) - 1
    gradient = np.zeros_like(function)

    gradient[1:N, 1:M, 1:L] = DDXi(function, N, M, L, hx) + DDEt(function, N, M, L, hy) + DDZt(function, N, M, L, hz)

    return gradient


# Central differencing for second derivatives along X direction
def DDXi(inpFld, Nz, Ny, Nx, hx):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nz, 1:Ny, 1:Nx] = (inpFld[1:Nz, 1:Ny, 0:Nx-1] - 2.0*inpFld[1:Nz, 1:Ny, 1:Nx] + inpFld[1:Nz, 1:Ny, 2:Nx+1])/(hx*hx)

    return outFld[1:Nz, 1:Ny, 1:Nx]


# Central differencing for second derivatives along Y direction
def DDEt(inpFld, Nz, Ny, Nx, hy):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nz, 1:Ny, 1:Nx] = (inpFld[1:Nz, 0:Ny-1, 1:Nx] - 2.0*inpFld[1:Nz, 1:Ny, 1:Nx] + inpFld[1:Nz, 2:Ny+1, 1:Nx])/(hy*hy)

    return outFld[1:Nz, 1:Ny, 1:Nx]


# Central differencing for second derivatives along Z direction
def DDZt(inpFld, Nz, Ny, Nx, hz):
    outFld = np.zeros_like(inpFld)
    outFld[1:Nz, 1:Ny, 1:Nx] = (inpFld[0:Nz-1, 1:Ny, 1:Nx] - 2.0*inpFld[1:Nz, 1:Ny, 1:Nx] + inpFld[2:Nz+1, 1:Ny, 1:Nx])/(hz*hz)

    return outFld[1:Nz, 1:Ny, 1:Nx]


# Pressure BCs for P
def imposePBCs(P):
    # Neumann boundary conditions
    # Left wall
    P[:, :, 0] = P[:, :, 2]

    # Right wall
    P[:, :, -1] = P[:, :, -3]

    # Front wall
    P[:, 0, :] = P[:, 2, :]

    # Back wall
    P[:, -1, :] = P[:, -3, :]

    # Bottom wall
    P[0, :, :] = P[2, :, :]

    # Top wall
    P[-1, :, :] = P[-3, :, :]

    return P


main()
