#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import json

import matplotlib

matplotlib.use('Agg') # For headless use

import matplotlib.pyplot as plt

import os

outputdir = sys.argv[1]

results = [('Backprop\n(1048576)', {'rodinia': 52065, 'old': 29276, 'new': 21456}),
           ('K-means\n(204800)', {'rodinia': 733786, 'old': 993759, 'new': 575838}),
           ('K-means\n(kdd_cup)', {'rodinia': 1484170, 'old': 896763, 'new': 688444}),
            ]

colours = { 'rodinia': '#cccccc',
            'old': '#00bfbf',
            'new': '#bf00bf' }

def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,
                1.02*height+10,
                '%d' % int(height),
                ha='center', va='bottom')

def plot(benchmark, results):
    filename = '{}/{}.pdf'.format(outputdir, benchmark)
    print ('Building {}...'.format(filename))

    N = len(results)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.20        # the width of the bars
    rects = {}

    fig, ax = plt.subplots()
    ax.set_ylabel('Runtime (ms)')
    ax.set_xlabel('Benchmark')
    plt.xlim([min(ind) - 0.25, max(ind) + 0.75])

    # add some text for labels, title and axes ticks
    ax.set_xticks(ind + width*1.5)
    ax.set_xticklabels(map(lambda (x,y): x ,results))

    i = 0
    for what in ['rodinia', 'old', 'new']:
        runtimes = [ dataset[what]/1000 for (_, dataset) in results ]
        rects[what] = ax.bar(ind+i*width, runtimes, width, color=colours[what],
                             align='center')
        i += 1

    ax.legend((rects['rodinia'], rects['old'], rects['new']),
              ('Rodinia', 'Segmented Scans', 'Segmented Reductions'),
              loc=2)

    autolabel(ax,rects['rodinia'])
    autolabel(ax,rects['old'])
    autolabel(ax,rects['new'])

    plt.rc('text')
    plt.savefig(filename, bbox_inches='tight')

plot('benchmarks', results)
