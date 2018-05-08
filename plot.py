import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import brewer2mpl
import math
import numpy as np
from time import sleep
import glob
import os

from baselines.a2c.utils import make_path

fontsize = 14
plt.style.use('ggplot')

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors


def load(filename):
    print(filename)
    data = []
    with open(filename) as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if i == 0:
                i += 1
                continue
            d = line.strip().split(";")
            if d[-1] == '':
                d = d[:-1]
            data.append(np.array(d).astype(np.float))
    return data


class DataPoint:

    def __init__(self, x, y):
        self.x = x
        self.y = y


def plot(path, title, data, smooth=10):
    print(title)

    color = '#1f77b4'
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
    label = "A2C"

    # Create datapoints
    points = []
    for d in data:
        for point in d:
            points.append(DataPoint(point[2], point[3]))

    # Sort data points
    points.sort(key=lambda x: x.x, reverse=True)

    x = []
    y = []
    step_x = []
    step_y = []
    ymin = []
    ymax = []

    for point in points:
        step_x.append(point.x)
        step_y.append(point.y)
        if len(step_x) == smooth:
            mean_y = np.mean(step_y)
            mean_x = np.mean(step_x)
            y.append(mean_y)
            x.append(mean_x)
            std_dev = np.std(step_y)
            ymin.append(mean_y - std_dev)
            ymax.append(mean_y + std_dev)
            step_x.clear()
            step_y.clear()

    plt.plot(x, y, linewidth=1, color=color)
    plt.fill_between(x, ymax, ymin, color=color, alpha=0.3)
    plt.title(title, fontsize=fontsize)
    plt.ylabel('Score')
    plt.xlabel('Steps')

    #handles, labels = plt.get_legend_handles_labels()
    #plt.legend(handles, labels, loc='upper center', ncol=2, fontsize=fontsize)

    plt.savefig(os.path.join(path, title + '.pdf'))


def main():

    for experiment_folder in glob.iglob('./results/*/'):
        title = experiment_folder.split('/')[-2].replace('-', ' ').title()
        path = os.path.join(experiment_folder, 'plots/')
        data = []
        for experiment_log in glob.iglob(os.path.join(experiment_folder, 'logs/*.log')):
            experiment_data = load(experiment_log)
            data.append(experiment_data)
        make_path(path)
        plot(path, title, data)
        plt.clf()


if __name__ == '__main__':
    main()

