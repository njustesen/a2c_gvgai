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
import argparse

from baselines.a2c.utils import make_path

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
            while '' in d:
                d.remove('')
            data.append(np.array(d).astype(np.float))
    return data


class DataPoint:

    def __init__(self, x, y, d=None):
        self.x = x
        self.y = y
        self.d = d


def plot_mixed(path, title, titles, datasets, smooth=10, fontsize=14):
    print(title)

    colors = ['#1f77b4', '#d62728', '#27d628', '#d627d6']

    fig, ax = plt.subplots()

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))

    i = 0
    for data in datasets:
        color = colors[i]

        # Create datapoints
        points = []
        for d in data:
            for point in d:
                if len(point) < 8:
                    points.append(DataPoint(point[2], point[3]))
                else:
                    points.append(DataPoint(point[2], point[3], d=point[7]))

        # Sort data points
        points.sort(key=lambda x: x.x, reverse=False)

        x = []
        y = []
        d = []
        step_x = []
        step_y = []
        step_d = []
        ymin = []
        ymax = []
        dmin = []
        dmax = []

        for point in points:
            step_x.append(point.x)
            step_y.append(point.y)
            if len(x) == 0 and smooth > 1 and False:
                x.append(point.x)
                y.append(point.y)
                ymin.append(point.y)
                ymax.append(point.y)
                if point.d is not None:
                    d.append(point.d)
                    dmin.append(point.d)
                    dmax.append(point.d)
            if point.d is not None:
                step_d.append(point.d)
            if len(step_x) == smooth:
                mean_y = np.mean(step_y)
                mean_x = np.mean(step_x)
                if point.d is not None:
                    mean_d = np.mean(step_d)
                    d.append(mean_d)
                    std_dev_d = np.std(step_d)
                    dmin.append(mean_d - std_dev_d)
                    dmax.append(mean_d + std_dev_d)
                y.append(mean_y)
                x.append(mean_x)
                std_dev = np.std(step_y)
                ymin.append(mean_y - std_dev)
                ymax.append(mean_y + std_dev)
                step_x.clear()
                step_y.clear()
                step_d.clear()

        lns1 = ax.plot(x, y, linewidth=1, color=color, label=titles[i])
        ax.fill_between(x, ymax, ymin, color=color, alpha=0.3)
        plt.title(title, fontsize=fontsize)
        ax.set_ylabel('Score')
        ax.set_xlabel('Steps')
        i += 1

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(path, title + '.pdf'))


def plot(path, title, data, smooth=10, fontsize=14, multiple=False):
    print(title)

    color = '#1f77b4'
    color_d = '#d62728'

    fig, ax1 = plt.subplots()

    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
    label = "A2C"

    # Create datapoints
    points = []
    if multiple:
        for d in data:
            for point in d:
                if len(point) < 8:
                    points.append(DataPoint(point[2], point[3]))
                else:
                    points.append(DataPoint(point[2], point[3], d=point[7]))
    else:
        for point in data:
            if len(point) < 8:
                points.append(DataPoint(point[2], point[3]))
            else:
                points.append(DataPoint(point[2], point[3], d=point[7]))

    # Sort data points
    points.sort(key=lambda x: x.x, reverse=False)

    x = []
    y = []
    d = []
    step_x = []
    step_y = []
    step_d = []
    ymin = []
    ymax = []
    dmin = []
    dmax = []

    for point in points:
        step_x.append(point.x)
        step_y.append(point.y)
        if len(x) == 0 and smooth > 1 and False:
            x.append(point.x)
            y.append(point.y)
            ymin.append(point.y)
            ymax.append(point.y)
            if point.d is not None:
                d.append(point.d)
                dmin.append(point.d)
                dmax.append(point.d)
        if point.d is not None:
            step_d.append(point.d)
        if len(step_x) == smooth:
            mean_y = np.mean(step_y)
            mean_x = np.mean(step_x)
            if point.d is not None:
                mean_d = np.mean(step_d)
                d.append(mean_d)
                std_dev_d = np.std(step_d)
                dmin.append(mean_d - std_dev_d)
                dmax.append(mean_d + std_dev_d)
            y.append(mean_y)
            x.append(mean_x)
            std_dev = np.std(step_y)
            ymin.append(mean_y - std_dev)
            ymax.append(mean_y + std_dev)
            step_x.clear()
            step_y.clear()
            step_d.clear()

    lns1 = ax1.plot(x, y, linewidth=1, color=color, label="Score")
    lns = lns1
    ax1.fill_between(x, ymax, ymin, color=color, alpha=0.3)
    ax2 = None
    ylabel_color = 'black'
    if len(d) > 0:
        ax2 = ax1.twinx()
        lns2 = ax2.plot(x, d, linewidth=1, color=color_d, label="Difficulty")
        lns += lns2
        ax2.set_ylabel('Difficulty', color=color_d)
        ylabel_color = color
        ax2.set_ylim([0, 1])
        #ax2.axis('off')
        ax2.grid(False)
        #ax1.fill_between(x, dmax, dmin, color=color_d, alpha=0.3)
    plt.title(title, fontsize=fontsize)
    ax1.set_ylabel('Score', color=ylabel_color)
    ax1.set_xlabel('Steps')

    labs = [l.get_label() for l in lns]
    if ax2 is not None:
        ax1.legend(lns, labs, loc=0)

    #handles, labels = plt.get_legend_handles_labels()
    #plt.legend(handles, labels, loc='upper center', ncol=2, fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(path, title + '.pdf'))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smooth', help='How many points to smooth', type=int, default=10)
    parser.add_argument('--font-size', help='Font size on plots', type=int, default=14)
    args = parser.parse_args()

    # Main plot for each experiment
    for experiment_folder in glob.iglob('./results/*/'):
        title = experiment_folder.split('/')[-2].replace('-', ' ').title()
        title = title.replace('Pcg', 'PCG').replace('Ls ', '')
        path = os.path.join(experiment_folder, 'plots/')
        make_path(path)
        data = []
        i = 0
        for experiment_log in glob.iglob(os.path.join(experiment_folder, 'logs/*.log')):
            i += 1
            experiment_data = load(experiment_log)
            experiment_title = title + " " + str(i)
            data.append(experiment_data)
            plot(path, experiment_title, experiment_data, smooth=args.smooth, fontsize=args.font_size, multiple=False)
            plt.clf()
        plot(path, title, data, smooth=args.smooth, fontsize=args.font_size, multiple=True)
        plt.clf()

    # Mixed plot for each experiment
    titles = []
    datasets = []
    for experiment_folder in glob.iglob('./results/*pcg-random*/'):
        title = experiment_folder.split('/')[-2].replace('-', ' ').title()
        title = title.replace('Pcg', 'PCG').replace('Ls ', '')
        data = []
        for experiment_log in glob.iglob(os.path.join(experiment_folder, 'logs/*.log')):
            experiment_data = load(experiment_log)
            data.append(experiment_data)
        datasets.append(data)
        titles.append(title)
    if len(titles) > 0:
        path = './plots/'
        make_path(path)
        plot_mixed(path, "PCG with Fixed Difficulty", titles, datasets, smooth=args.smooth, fontsize=args.font_size)
        plt.clf()


if __name__ == '__main__':
    main()

