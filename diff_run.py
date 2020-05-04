import subprocess as sp
sp.call(["nvcc","diff_fxns.cu","-o","df.exe"])
i = 1
while(i <= 16):
    sp.call(["./df.exe",str(i)])
    i *= 2