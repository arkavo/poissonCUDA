import subprocess as sp

threads = [1,2,4,8]
data = [10,15,20,25,30]
for i in threads:
    for j in data:
        for k in data:
            for l in data:
                sp.call(["./pCUDA.exe",str(i),str(j),str(k),str(l)])
