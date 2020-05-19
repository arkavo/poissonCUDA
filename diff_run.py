import subprocess as sp
import readline
sp.call(["nvcc","diff_fxns.cu","-o","df.exe"])

i = 1
while(i <= 2**7):
    sp.call(["./df.exe",str(i)])
    x = readline.get_line_buffer()
    i *= 2

print(x)


