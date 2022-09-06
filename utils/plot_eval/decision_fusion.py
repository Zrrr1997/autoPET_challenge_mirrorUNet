import numpy as np
import argparse
import os
from matplotlib import pyplot as plt


def acc(preds, gt):
	assert len(preds) == len(gt)
	return np.sum(preds == gt) / len(gt)

parser = argparse.ArgumentParser(description='Late fusion parser.')
parser.add_argument('--dirs', type=str, nargs='+', default=None,
                     help='Directories to evaluate.')
parser.add_argument('--plot_logit_pdf', type=str, default=None,
                     help='Directory to plot prediction logits and exit.')

args = parser.parse_args()
dirs = args.dirs

labels = np.load(os.path.join(dirs[0], 'labels.npy'))
# Check if all directories have the same GT labels
for d in dirs:
	labels_check = np.load(os.path.join(d, 'labels.npy'))
	assert np.all(labels == labels_check)


# Voting Fusion
fp_fn = []
best_fn = 200

# Find best threshold
for th in np.arange(0.025, 0.975, 0.025):
	incorrect = []
	fp = []
	fn = []
	incorrect_labels = []

	curr_preds = np.load(os.path.join(dirs[0], 'outputs.npy'))
	raw_preds = curr_preds.copy()
	curr_preds = (curr_preds > th).astype(np.uint32)
	preds = curr_preds

	for d in dirs[1:]:
		curr_preds = np.load(os.path.join(d, 'outputs.npy'))

		curr_preds = (curr_preds > th).astype(np.uint32)
		preds = np.logical_or(preds, curr_preds)
	for i, (x, y) in enumerate(zip(preds, labels)):
		if x != y:
			incorrect.append(i)
			if y == 1:
				fn.append(i)
				if abs(th - 0.3) < 0.001:
					print(i)
			else:
				fp.append(i)
			incorrect_labels.append(int(y))
	if args.plot_logit_pdf is not None and abs(th - 0.05) < 0.001:
		plt.hist(raw_preds, label='All predictions.')
		error_logits = []
		for err_labels in incorrect_labels:
			error_logits.append(raw_preds[err_labels])
		plt.hist(error_logits, label='Incorrect predictions.')
		plt.legend()
		plt.xlabel('Sigmoid Output')
		plt.ylabel('Frequency')

		os.makedirs(args.plot_logit_pdf, exist_ok=True)
		plt.savefig(os.path.join(args.plot_logit_pdf, 'sigmoid_hist.png'))
	if abs(th - 0.05) < 0.001:
		print('FN:', len(fn), 'th:', round(th, 3), 'acc:', round(acc(preds, labels), 3), 'FP:', len(fp))
	fp_fn.append((len(fp), len(fn)))
	if best_fn >= len(fn):
		best_fn = len(fn)
		best_th = th
		best_acc = acc(preds, labels)
		best_fp = len(fp)
	#if th > 0.5:
		#break

#print('FN:', best_fn, 'th:', round(best_th, 3), 'acc:', round(best_acc, 3), 'FP:', best_fp)
