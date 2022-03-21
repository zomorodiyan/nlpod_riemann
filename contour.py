import matplotlib.pyplot as plt
import numpy as np

# Data
diry = "Results/Euler_4/"
x = np.genfromtxt(diry + "X.csv", delimiter=",")
y = np.genfromtxt(diry + "Y.csv", delimiter=",")
z = np.genfromtxt(diry + "P200.csv", delimiter=",")

# plot.
fig, ax = plt.subplots(ncols=1)
fig.set_size_inches(6.5, 5)
cs = ax.contourf(x, y, z, levels=200, cmap="coolwarm")
cbar = fig.colorbar(cs)
ax.set_title("Lax-Liu Conf.12, top_right_initial=0.4")
plt.ylabel("x")
plt.xlabel("y")
ax.grid(c="k", ls="-", alpha=0.08)

# plt.savefig('Plots/Euler.png')
plt.show()
