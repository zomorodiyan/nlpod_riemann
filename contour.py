import matplotlib.pyplot as plt
import numpy as np

# Data
diry = "Results/npz/"
x = np.load(diry + "X.npy")
y = np.load(diry + "Y.npy")
z = np.load(diry + "bc_1/P40.npy")
print(x.shape)
print(y.shape)
print(z.shape)

# plot.
fig, ax = plt.subplots(ncols=1)
plt.rcParams.update({"font.size": 12})
fig.set_size_inches(6.5, 5)
cs = ax.contourf(x, y, z, levels=200, cmap="coolwarm")
cbar = fig.colorbar(cs)
ax.set_title("Lax-Liu Conf.12, top_right_initial=0.4")
plt.ylabel("x")
plt.xlabel("y")
ax.grid(c="k", ls="-", alpha=0.08)

# plt.savefig('Plots/Euler.png')
plt.show()
