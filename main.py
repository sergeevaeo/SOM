from minisom import MiniSom

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

# чтение данных из файла
data = pd.read_csv('stock_prices.txt',
                   names=['open', 'high', 'low',
                          'close', 'adjClose', 'target'],
                   sep='\t+')
t = data['target'].values
data = data[data.columns[:-1]]

# нормализация данных
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Инициализация и тренировка SOM размером 12х12
som = MiniSom(12, 12, data.shape[1], sigma=1.5, learning_rate=.5, activation_distance='euclidean',
              topology='hexagonal', neighborhood_function='gaussian', random_seed=10)

som.train(data, 1000, verbose=True)

# положение нейронов на евклидовой плоскости
xx, yy = som.get_euclidean_coordinates()
# карта расстояний весов
umatrix = som.distance_map()
print(umatrix)
# веса
weights = som.get_weights()

# отрисовка карты
f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.set_aspect('equal')

# добавление шестиугольников и их цветов на карту
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)] * np.sqrt(3) / 2
        hexes = RegularPolygon((xx[(i, j)], wy),
                               numVertices=6,
                               radius=.95 / np.sqrt(3),
                               facecolor=cm.Reds(umatrix[i, j]),
                               alpha=.4,
                               edgecolor='gray')
        ax.add_patch(hexes)

# добавление меток
markers = ['o', '+', 'x']
colors = ['C0', 'C1', 'C2']
for cnt, x in enumerate(data):
    # получаем победителя
    w = som.winner(x)
    # преобразование в евклидовы координаты
    wx, wy = som.convert_map_to_euclidean(w)
    wy = wy * np.sqrt(3) / 2
    # расставление меток
    plt.plot(wx, wy,
             markers[t[cnt] - 1],
             markerfacecolor='None',
             markeredgecolor=colors[t[cnt] - 1],
             markersize=12,
             markeredgewidth=2)

# график
xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange - .5, xrange)
plt.yticks(yrange * np.sqrt(3) / 2, yrange)

# цветная полоса
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Reds,
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 20
cb1.ax.set_ylabel('расстояние между нейронами по соседству',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

# описание
legend_elements = [Line2D([0], [0], marker='o', color='C0', label='Apple',
                          markerfacecolor='w', markersize=16, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='+', color='C1', label='Google',
                          markerfacecolor='w', markersize=16, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='x', color='C2', label='Microsoft',
                          markerfacecolor='w', markersize=16, linestyle='None', markeredgewidth=2)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left',
          borderaxespad=0., ncol=3, fontsize=16)

plt.savefig('resulting_images/stock.png')
plt.show()
