import subprocess as sp

sp.call(["nvcc","poisson3d.cu","-o","p3d.exe"])
sp.call(["./p3d.exe"])