#!/usr/bin/env python

import numpy as np
import sys
import json

import matplotlib

matplotlib.use('Agg') # For headless use

import matplotlib.pyplot as plt

import os

outputdir = sys.argv[1]

def mk_data_sets(n):
    return [ ('2pow' + str(k), '2pow' + str(n-k)) for k in range(0,n+1,2) ]

common_data_sets={"2pow26": mk_data_sets(26),
                  "2pow18": mk_data_sets(18)
}

benchmarks=[('sum', 'Segmented sum', 'i32', common_data_sets)]

variants=[('segmented_auto', 'segmented', "Automatic"),
          ('segmented_large', 'segmented', "Large"),
          ('segmented_small', 'segmented', "Small"),
          ('segmented_map_with_loop', 'segmented', "Map-with-loop"),
          ('segmented_scan', 'segmented_scan', "Segmented scan")]

group_size=128

markers=['x', 'p', 'o', 'v', '+', 'D']

def data_file(benchmark, variant, group_size):
    variant = '_' + variant if variant != None else ''
    return 'results/{}{}_groupsize_{}.json'.format(benchmark, variant, group_size)

for benchmark, benchmark_name, benchmark_data, benchmark_data_sets in benchmarks:
    for work in benchmark_data_sets:
        fig, ax = plt.subplots()
        ax.set_ylabel('Runtime (ms)')
        ax.set_xlabel('Data set')

        xticks = []
        for (num_segments,segment_size) in benchmark_data_sets[work]:
            xticks += ['[{}][{}]'.format(num_segments,segment_size)]

        ylims=[]
        for marker, (variant, variant_file, variant_desc) in zip(markers, variants):
            f_json = json.load(open(data_file(benchmark, variant, group_size)))
            results = f_json['benchmarks/{}_{}.fut'.format(benchmark, variant_file)]

            xs=[]
            ys=[]
            i = 0
            for (num_segments,segment_size) in benchmark_data_sets[work]:
                dk = 'data/{}_{}_{}'.format(benchmark_data, num_segments, segment_size)
                r = results['datasets'][dk]
                if type(r) is dict:
                    ms = np.mean(r['runtimes'])/1000.0
                    if ms > 0.01: # sanity check
                        xs += [i]
                        ys += [ms]
                i += 1
            ylims += [np.max(ys)]
            ax.plot(xs,ys,label=variant_desc,marker=marker)

        grey='#aaaaaa'

        ylims.sort()
        ax.set_ylim([0,ylims[-2]*1.2])
        ax.yaxis.grid(color=grey,zorder=0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        ax.set_xticks(1+np.arange(len(xticks)))
        ax.set_xticklabels(xticks, rotation=-45)

        plt.rc('text')
        plt.savefig('{}/{}_{}_group_size_{}.pdf'.format(outputdir, benchmark, work, group_size),
                    bbox_inches='tight')
