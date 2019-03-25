import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(-3, 3), ylim=(-1, 1))

x = np.linspace(-3, 3, 91)
t = np.linspace(0, 25, 30)
y = np.linspace(-3, 3, 91)
X3, Y3, T3 = np.meshgrid(x, y, t)
sinT3 = np.sin(2*np.pi*T3/T3.max(axis=2)[..., np.newaxis])
G = (X3**2 + Y3**2)*sinT3

cax = ax.pcolormesh(x, y, G[:-1, :-1, 0], vmin=-1, vmax=1, cmap='Blues')
fig.colorbar(cax)

def animate(i):
     cax.set_array(G[:-1, :-1, i].flatten())

anim = FuncAnimation(
    fig, animate, interval=100, frames=len(t)-1)

plt.draw()
plt.show()
