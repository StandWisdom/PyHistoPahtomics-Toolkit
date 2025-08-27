import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_tissuemap(saveDir, name):
    result_dir = saveDir
    result_df = pd.read_excel(os.path.join(result_dir, name+'_pred-matrix.xlsx'))
    xy = np.array(result_df.iloc[:, :2])
    pred = np.array(result_df.iloc[:, 2:])

    tilesize = 256
    step = 1
    wsi_pyramid = [2, 4, 8]
    ratio = tilesize * step * wsi_pyramid[0]

    h_ = np.min(xy[:, 1])
    w_ = np.min(xy[:, 0])
    h_min = h_
    w_min = w_
    h_max = np.max(xy[:, 1])
    w_max = np.max(xy[:, 0])
    h = int((h_max - h_) // ratio + 1)
    w = int((w_max- w_) // ratio + 1)
    canvas = np.zeros([h, w])

    for i in range(xy.shape[0]):
        x = (xy[i, 1] - h_) // ratio
        y = (xy[i, 0] - w_) // ratio
        pred_ = pred[i, :]
        label = np.argmax(pred_)

        canvas[x, y] = label + 1
        # if int(label) in [0]:
        #     label = 0
        # elif int(label) in [2,3,4]:
        #     label = 1
        # elif int(label) in [1]:
        #     label = 2
        # elif int(label) in [5,6,8,9]:
        #     label = 3
        # elif int(label) in [11]:
        #     label = 4
        # else:
        #     label = 5

    # 创建自定义颜色映射
    colors = ['white',
              'red', 'orange', 'blue',
              'lime', 'sienna', 'green', 'gray',
              'yellow', 'cyan', 'pink', 'fuchsia', 'black']
    values = np.array(range(0, len(colors), 1))
    cmap = ListedColormap(colors)
    plt.imshow(canvas, cmap=cmap, interpolation='nearest', vmin=min(values), vmax=max(values))
    plt.colorbar()
    plt.axis('off')
    figDir = './results/tsmap'
    if not os.path.exists(figDir):
        os.makedirs(figDir)
    plt.savefig(os.path.join(figDir,name+'_tsmap.pdf'))
    plt.show()

srcDir = r'E:\pathomics2024\PyHistomics\results\pumch'
namesRef = list(next(os.walk(r'L:\pancreatic cancer\PUMCH\PDAC-annotation'))[1])
names = list(next(os.walk(srcDir))[2])
for name in names:
    name = name.split('_pred-matrix.xlsx')[0]
    if name in namesRef:
        plot_tissuemap(srcDir,name)









