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
            if d[-1] == '':
                d = d[:-1]
            data.append(np.array(d).astype(np.float))
    return data


class DataPoint:

    def __init__(self, x, y, d=None):
        self.x = x
        self.y = y
        self.d = d


def plot_mixed(path, title, titles, datasets, smooth=10, fontsize=14):
    print(title)

    colors = ['#CF8E01', '#D54949', '#49D5C6',  '#4954D5']

    fig, ax = plt.subplots()

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
    matplotlib.rcParams.update({'font.size': fontsize})

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

        if len(x) == 0:
            return

        lns1 = ax.plot(x, y, linewidth=2, color=color, label=titles[i])
        ax.fill_between(x, ymax, ymin, color=color, alpha=0.2)
        plt.title(title, fontsize=fontsize)
        ax.set_ylabel('Score')
        ax.set_xlabel('Steps')
        ax.set_xlim([0, np.max(x) + np.max(x) * 0.005])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

        i += 1

    handles, labels = ax.get_legend_handles_labels()
    import operator
    hl = sorted(zip(handles, labels),
                key=operator.itemgetter(1))
    handles, labels = zip(*hl)
    #handles = handles[::-1]
    #labels = labels[::-1]
    ax.legend(handles, labels, loc='upper left', prop={'size': fontsize-4})
    ax.xaxis.get_offset_text().set_fontsize(fontsize)

    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.85, top=0.9, bottom=0.15)
    fig.savefig(os.path.join(path, title + '.pdf'))


def plot(path, title, data, smooth=10, fontsize=14, multiple=False, ymin_lim=None, ymax_lim=None):
    print(title)

    color = '#1f77b4'
    color_d = '#d62728'

    fig, ax1 = plt.subplots()

    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 4), labelsize=fontsize)
    matplotlib.rcParams.update({'font.size': fontsize})
    plt.rc('font', size=fontsize)
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
        if len(x) == 0 and smooth > 1 and True:
            x.append(point.x)
            y.append(point.y)
            ymin.append(point.y)
            ymax.append(point.y)
            if point.d is not None:
                d.append(0)
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

    if len(x) == 0:
        return

    lns1 = ax1.plot(x, y, linewidth=2, color=color, label="Score")
    lns = lns1
    ax1.fill_between(x, ymax, ymin, color=color, alpha=0.2)
    ax2 = None
    ylabel_color = 'black'
    if len(d) > 0:
        ax2 = ax1.twinx()
        lns2 = ax2.plot(x, d, linewidth=2, color=color_d, label="Difficulty")
        lns += lns2
        ax2.set_ylabel('Difficulty', color=color_d)
        ylabel_color = color
        ax2.set_ylim([0, 1])
        #ax2.set_xlim([0, np.max(x)])
        ax2.set_xlim([0, int(40e6) + int(40e6) * 0.001])
        ax2.xaxis.get_offset_text().set_fontsize(fontsize)
        #ax2.axis('off')
        ax2.grid(False)
        #ax1.fill_between(x, dmax, dmin, color=color_d, alpha=0.3)
        for item in ([ax2.title,
                      ax2.xaxis.label,
                      ax2.yaxis.label] +
                     ax2.get_xticklabels() +
                     ax2.get_yticklabels()):
            item.set_fontsize(fontsize)
    #plt.title(title, fontsize=fontsize)
    ax1.set_ylabel('Score', color=ylabel_color)
    ax1.set_xlabel('Steps')
    #ax1.set_xlim([0, np.max(x)])
    ax1.set_xlim([0, int(40e6) + int(40e6) * 0.001])
    if ymin is not None and ymax is not None:
        ax1.set_ylim([ymin_lim, ymax_lim])

    for item in ([ax1.title,
                    ax1.xaxis.label,
                    ax1.yaxis.label] +
                    ax1.get_xticklabels() +
                    ax1.get_yticklabels()):
        item.set_fontsize(fontsize)

    ax1.xaxis.get_offset_text().set_fontsize(fontsize)

    labs = [l.get_label() for l in lns]
    if ax2 is not None:
        ax1.legend(lns, labs, loc="lower right", prop={'size': fontsize-4})

    #handles, labels = plt.get_legend_handles_labels()
    #plt.legend(handles, labels, loc='upper center', ncol=2, fontsize=fontsize)
    fig.tight_layout()
    # FROGS: fig.subplots_adjust(left=0.17, right=0.82, top=0.95, bottom=0.18)
    fig.subplots_adjust(left=0.19, right=0.83, top=0.95, bottom=0.18)

    fig.savefig(os.path.join(path, title + '.pdf'))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smooth', help='How many points to smooth', type=int, default=200)
    parser.add_argument('--font-size', help='Font size on plots', type=int, default=24)
    args = parser.parse_args()

    # Main plot for each experiment

    for experiment_folder in glob.iglob('./results/solarfox-ls-pcg-progressive-fixed/'):
        title = experiment_folder.split('/')[-2].replace('-', ' ').title()
        title = title.replace('Pcg', 'PCG').replace('Ls ', '')
        path = os.path.join(experiment_folder, 'plots/')
        data = []
        i = 0
        for experiment_log in glob.iglob(os.path.join(experiment_folder, 'logs/*.log')):
            i += 1
            experiment_data = load(experiment_log)
            experiment_title = title + " " + str(i)
            data.append(experiment_data)
            #plot(path, experiment_title, experiment_data, smooth=args.smooth, fontsize=args.font_size, multiple=False)
            plt.clf()
        make_path(path)
        title = "Progressive PCG in Solarfox"
        plot(path, title, data, smooth=args.smooth, fontsize=args.font_size, multiple=True, ymin_lim=None, ymax_lim=None)
        plt.clf()


    # Mixed plot for each experiment
    '''
    titles = []
    datasets = []
    for experiment_folder in glob.iglob('./results/frogs-ls-pcg-random-*/'):
        title = experiment_folder.split('/')[-2].replace('-', ' ').title()
        title = title.replace('Pcg', 'PCG').replace('Ls ', '')
        title = title.replace('Random ', '')
        title = title.replace('3', '.3')
        title = title.replace('5', '.5')
        title = title.replace('7', '.7')
        title = title.replace('10', '1')
        title = title.replace('Zelda ', '')
        data = []
        for experiment_log in glob.iglob(os.path.join(experiment_folder, 'logs/*.log')):
            experiment_data = load(experiment_log)
            data.append(experiment_data)
        datasets.append(data)
        titles.append(title)
    if len(titles) > 0:
        path = './plots/'
        make_path(path)
        plot_mixed(path, "PCG1 in Frogs", titles, datasets, smooth=args.smooth, fontsize=args.font_size)
        plt.clf()
    '''


if __name__ == '__main__':
    main()

