import subprocess as sp

threads = [1,2,4,8]
data = [10]
for i in threads:
    for j in data:
        sp.call(["./pCUDA.exe",str(i),str(j),str(j),str(j)])
