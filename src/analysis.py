import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data.csv",names=["Threads","X","Y","Z","Size","Execution Time"])

print(data)
for i in range(5):
    Y = data["Execution Time"].iloc[3*i:3*i+3]
    X = data["Size"].iloc[3*i:3*i+3]

    plt.plot(X,Y)
    plt.scatter(X,Y, marker='v', color='r')
plt.legend(["1 Thread","2 threads","4 threads","8 threads","16 Threads"])
plt.xlabel("Data Size")
plt.ylabel("Execution time(s)")
plt.title("Execution time comparison between data sizes and threads")
plt.xscale("log")
plt.yscale("log")
#plt.xscale([1000,8000,27000])
plt.savefig("figures/log_fig.png")
print(X,Y)
