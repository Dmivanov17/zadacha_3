import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def rastrigin(x1, x2):
    D = 2  # Размерность
    term1 = x1**2 - 10 * np.cos(2 * np.pi * x1)
    term2 = x2**2 - 10 * np.cos(2 * np.pi * x2)
    return 10 * D + (term1) + (term2)


x1_min, x1_max = -5.12, 5.12
x2_min, x2_max = -5.12, 5.12
test_point = (0.0, 0.0)
step = 0.1


x1 = np.arange(x1_min, x1_max + step, step)
x2 = np.arange(x2_min, x2_max + step, step)
X1, X2 = np.meshgrid(x1, x2)
Y = rastrigin(X1, X2)


test_value = rastrigin(test_point[0], test_point[1])


fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'Функция Растригина\nТестовая точка: {test_point}, f{test_point} = {test_value:.2f}', fontsize=14)


ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(X1, X2, Y, cmap=cm.viridis, linewidth=0, antialiased=True)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$y = f(x_1, x_2)$')
ax1.set_title('3D поверхность (изометрический вид)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)


ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(X1, X2, Y, cmap=cm.viridis, linewidth=0, antialiased=True)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('$y = f(x_1, x_2)$')
ax2.set_title('3D поверхность (вид сверху)')
ax2.view_init(elev=90, azim=0)  # Вид сверху
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)


x2_fixed = test_point[1]
y_x2_fixed = rastrigin(x1, x2_fixed)
ax3 = fig.add_subplot(223)
ax3.plot(x1, y_x2_fixed)
ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$y = f(x_1, x_{20})$')
ax3.set_title(f'График при $x_2 = {x2_fixed}$')
ax3.grid(True)


x1_fixed = test_point[0]
y_x1_fixed = rastrigin(x1_fixed, x2)
ax4 = fig.add_subplot(224)
ax4.plot(x2, y_x1_fixed)
ax4.set_xlabel('$x_2$')
ax4.set_ylabel('$y = f(x_{10}, x_2)$')
ax4.set_title(f'График при $x_1 = {x1_fixed}$')
ax4.grid(True)


plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.show()
