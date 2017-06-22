# Compare the curated results to jason's results
import my
import pandas
import numpy as np
import os
import scipy.optimize
import argparse

## Parse arguments
parser = argparse.ArgumentParser(description="""
    Run the kalman filter on an example whiskers dataset.
""")
parser.add_argument("results",
    help='Name of the results file containing labeled whiskers')
args = parser.parse_args()
results_filename = args.results

## Load curated results
curated_filename = os.path.expanduser(
    '~/mnt/nas2_home/whisker/test_bed/161215_KM91/curated')
curated_df = pandas.read_pickle(curated_filename)

# Load jason's results
#~ results_filename = os.path.expanduser(
    #~ '~/mnt/nas2_home/whisker/test_bed/161215_KM91/15000_frames_revised.pickle')
results = my.misc.pickle_load(results_filename)

# Make labels consistent
jres = results[['frame', 'seg', 'color_group']].set_index(['frame', 'seg'])
cres = curated_df.set_index(['frame', 'seg'])

# Join jason's results onto curated results
ares = cres.join(jres, rsuffix='_jason', how='inner')

# For right now drop the unlabeled ones in the curated dataset
ares = ares[ares.color_group != -1].copy()

# Count confusion matrix
confusion_matrix = ares.reset_index().pivot_table(
    index='color_group', columns='color_group_jason', values='frame', 
    aggfunc='count').fillna(0).astype(np.int)

# Assign
# j_labels will be Jason's labels for curated classes in sorted order
c_labels, j_labels = scipy.optimize.linear_sum_assignment(-confusion_matrix)

# Calculate performance
perf_whisker_l = []
n_hits = 0
for c_label, j_label in zip(c_labels, j_labels):
    n_hits += confusion_matrix.iloc[c_label, j_label]
    perf_whisker = confusion_matrix.iloc[c_label, j_label] / float(
        confusion_matrix.iloc[c_label].sum())
    perf_whisker_l.append(perf_whisker)
frac_correct = n_hits / float(len(ares))    
worst_whisker = np.min(perf_whisker_l)

print "Overall: %0.2f" % frac_correct
print "Worst whisker: %0.2f" % worst_whisker
print "Confusion matrix: "
print confusion_matrix