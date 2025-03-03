import torch
import matplotlib.pyplot as plt
import os

files= [f for f in os.listdir(os.getcwd()) if f.endswith("_all")]
f=files[0]

xpl=torch.load(os.path.join(os.getcwd(), f), map_location="cpu")

for i in range(xpl.shape[0]):
    x=xpl[i].topk(k=xpl.shape[1], sorted=True, largest=False)[0]
    plt.plot(x)

plt.savefig(os.path.join(os.getcwd(), "influence_plot.png"))
