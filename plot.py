import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import brewer2mpl
import math
import numpy as np
from time import sleep
import glob

from baselines.a2c.utils import make_path

fontsize = 14
plt.style.use('ggplot')

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors


def load(filename):
    print(filename)
    title = filename.split(".")[0]
    data = []
    with open(filename) as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if i == 0:
                i += 1
                continue
            d = line.strip().split(";")
            data.append(np.array(d).astype(np.float))
    return data


def plot(title, data, smooth=10):
    print(title)

    color = '#1f77b4'
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
    label = "A2C"
    x = []
    y = []
    smoothened_x = []
    smoothened_y = []
    smoothened_std = []
    ymin = []
    ymax = []
    for d in data:
        smoothened_x.append(d[2])
        smoothened_y.append(d[3])
        smoothened_std.append(d[4])
        if len(smoothened_x) == smooth:
            y.append(np.mean(smoothened_y))
            x.append(np.mean(smoothened_x))
            ymin.append(np.mean(smoothened_y) - np.mean(smoothened_std))
            ymax.append(np.mean(smoothened_y) + np.mean(smoothened_std))
            smoothened_x.clear()
            smoothened_y.clear()
            smoothened_std.clear()

    plt.plot(x, y, linewidth=1, color=color)
    plt.fill_between(x, ymax, ymin, color=color, alpha=0.3)
    plt.title(title, fontsize=fontsize)
    plt.ylabel('Score')
    plt.xlabel('Steps')

    #handles, labels = plt.get_legend_handles_labels()
    #plt.legend(handles, labels, loc='upper center', ncol=2, fontsize=fontsize)

    make_path('./plots')
    plt.savefig('./plots/' + title + '.pdf')

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():
    parser = arg_parser()
    args = parser.parse_args()

    for file in glob.iglob('./logs/a2c/*.log'):
        data = load(file)
        title = file.split('/')[-1].split('.')[0].replace('gvgai-','').replace('-v0','').replace('-', ' ').replace('_', ' ').replace('lg', '')
        plot(title, data)

if __name__ == '__main__':
    main()

