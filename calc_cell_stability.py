import numpy as np
import matplotlib.pyplot as plt

m = 0.01
theta = 30*np.pi/180
g = 9.81

a = np.arange(0, 10, 0.001)

fg = m*g
fn1 = 0.5*(fg/np.cos(theta) - m*a/np.sin(theta))
fn0 = fg/np.cos(theta) - fn1

fig, ax = plt.subplots(1)
ax.plot(a, fn0)
ax.plot(a, fn1)
plt.show()