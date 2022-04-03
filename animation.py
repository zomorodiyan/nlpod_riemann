import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable

diry = "Results/npz/"
x = np.load(diry + "X.npy")
y = np.load(diry + "Y.npy")
z = np.load(diry + "bc_5/P0.npy")

DATA = np.random.randn(800).reshape(10, 10, 8)

fig, ax = plt.subplots()
SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 14
HUGE_SIZE = 15
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes

fig.set_size_inches(6.5, 5)
vmin = 0
vmax = 2.5
cs = ax.contourf(x, y, z, levels=200, cmap="coolwarm", vmin=vmin, vmax=vmax)
fig.colorbar(
    ScalarMappable(norm=cs.norm, cmap=cs.cmap),
    ticks=np.arange(vmin, vmax + 0.01, 0.5),
)

ax.set_title("Lax-Liu Conf.12: top_right_initial=0.4")
plt.ylabel("x")
plt.xlabel("y")
ax.grid(c="k", ls="-", alpha=0.08)


def animate(i):
    z = np.load(diry + "bc_4/P" + str(i) + ".npy")
    cs = ax.contourf(x, y, z, levels=200, cmap="coolwarm", vmin=vmin, vmax=vmax)


interval = 0.01  # in seconds
ani = animation.FuncAnimation(fig, animate, 50, interval=interval * 1e3, blit=False)
plt.show()
print("almost")
ani.save("./Plots/euler.gif", writer="imagemagick", fps=30)
print("done!")
