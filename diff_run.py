import subprocess as sp
sp.call(["nvcc","diff_fxns.cu","-o","df.exe"])
sp.call("./df.exe")