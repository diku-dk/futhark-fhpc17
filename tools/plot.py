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

def mk_data_sets(n):
    return [ ('2pow' + str(k), '2pow' + str(n-k)) for k in range(0,n+1,2) ]

common_data_sets={"2pow26": mk_data_sets(26),
                  "2pow18": mk_data_sets(18)
}

benchmarks=[('sum', 'Segmented sum', 'i32', common_data_sets),
            ('mss', 'MSS', 'i32', common_data_sets),
            ('index_of_max', 'Index of maximum', 'i32', common_data_sets),
            ('blackscholes', 'Black-Scholes', 'blackscholes', common_data_sets)]

variants=[('segmented_large', '_segmented', "Large segments"),
          ('segmented_small', '_segmented', "Small segments"),
          ('segmented_map_with_loop', '_segmented', "Sequential segments"),
          ('segmented_scan', '_segmented_scan', "Segmented scan"),
          ('segmented_auto', '_segmented', "Automatic"),
]

markers=['x', 'p', '|', 'v', '*', 'D']

def data_file(benchmark, variant, group_size):
    variant = '_' + variant if variant != None else ''
    return 'results/{}{}_groupsize_{}.json'.format(benchmark, variant, group_size)

def ylimit(benchmark, work, ylims):
    limits = {'sum': {'2pow18': 4,
                      '2pow26': 32},

              'index_of_max': {'2pow18': 4,
                               '2pow26': 50},

              'mss': {'2pow18': 6,
                      '2pow26': 65},

              'blackscholes': {'2pow18': 4,
                               '2pow26': 100}}
    try:
        return limits[benchmark][work]
    except KeyError:
        ylims.sort()
        return ylims[-2]*1.2

for benchmark, benchmark_name, benchmark_data, benchmark_data_sets in benchmarks:
    group_sizes = [128, 512, 1024] if benchmark == 'sum' else [512]
    for group_size in group_sizes:
        for work in benchmark_data_sets:
            filename = '{}/{}_{}_group_size_{}.pdf'.format(outputdir, benchmark, work, group_size)
            print ('Building {}...'.format(filename))

            fig, ax = plt.subplots()
            ax.set_ylabel('Runtime (ms)')
            ax.set_xlabel('Data set')

            xticks = []
            for (num_segments,segment_size) in benchmark_data_sets[work]:
                # Ugly hack to get superscript labels - could not get the pseudo-TeX to work.
                def powlabel(s):
                    base,exp = unicode(s).split('pow')
                    exp = exp.replace(u'0', u'⁰').replace(u'1', u'¹').replace(u'2', u'²').replace(u'4', u'⁴').replace(u'6', u'⁶').replace(u'8', u'⁸')
                    return base + exp
                xticks += [(u'[{}][{}]').format(powlabel(num_segments),powlabel(segment_size))]
            # Add non-segmented baseline.
            xs=[]
            ys=[]
            f_json = json.load(open(data_file(benchmark, None, group_size)))
            results = f_json['benchmarks/{}.fut'.format(benchmark)]
            i = 0
            for (num_segments,segment_size) in benchmark_data_sets[work]:
                dk = 'inputs/{}_{}'.format(benchmark_data, work)
                r = results['datasets'][dk]
                ms = np.mean(r['runtimes'])/1000.0
                xs += [i]
                ys += [ms]
                i += 1
            ax.plot(xs,ys,label='Non-segmented',marker='',linewidth=3,color='black',ls='dashed')

            # Add CUB version if it exists.
            try:
                f_json = json.load(open('results/{}_segmented_cub.json'.format(benchmark)))
                i = 0
                xs=[]
                ys=[]
                for (num_segments,segment_size) in benchmark_data_sets[work]:
                    dk = 'inputs/{}_{}_{}'.format(benchmark_data, num_segments, segment_size)
                    if len(f_json[dk]) > 0:
                        ms=float(f_json[dk])/1000
                        xs += [i]
                        ys += [ms]
                    i += 1
                ax.plot(xs,ys,label='CUB',marker='D',linewidth=3,color='#30d62a')
            except IOError:
                pass

            # Now add the segmented runtimes.
            ylims=[]
            for marker, (variant, variant_file, variant_desc) in zip(markers, variants):
                f_json = json.load(open(data_file(benchmark, variant, group_size)))
                results = f_json['benchmarks/{}{}.fut'.format(benchmark, variant_file)]

                xs=[]
                ys=[]
                i = 0
                for (num_segments,segment_size) in benchmark_data_sets[work]:
                    dk = 'inputs/{}_{}_{}'.format(benchmark_data, num_segments, segment_size)
                    r = results['datasets'][dk]
                    if type(r) is dict:
                        ms = np.mean(r['runtimes'])/1000.0
                        xs += [i]
                        ys += [ms]
                    i += 1
                ylims += [np.max(ys)]
                ax.plot(xs,ys,label=variant_desc,marker=marker,linewidth=3,markersize=15)

            grey='#aaaaaa'

            ylims.sort()
            ax.set_ylim([0,ylimit(benchmark, work, ylims)])
            ax.yaxis.grid(color=grey,zorder=0)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            ax.set_xticks(np.arange(len(xticks)))
            ax.set_xticklabels(xticks, rotation=-45,size='large')

            plt.rc('text')
            plt.savefig(filename, bbox_inches='tight')
