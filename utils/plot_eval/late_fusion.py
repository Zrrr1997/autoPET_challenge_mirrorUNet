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

args = parser.parse_args()
dirs = args.dirs



labels = np.load(os.path.join(dirs[0], 'labels.npy'))
# Check if all directories have the same GT labels
for d in dirs:
	labels_check = np.load(os.path.join(d, 'labels.npy'))
	assert np.all(labels == labels_check)


# Sum Fusion
preds = np.load(os.path.join(dirs[0], 'outputs.npy'))


th = 0.5
curr_preds = (preds > th).astype(np.uint32)
print(acc(curr_preds, labels), dirs[0])
for d in dirs[1:]:
	curr_preds = np.load(os.path.join(d, 'outputs.npy'))
	preds += curr_preds
	curr_preds = (curr_preds > th).astype(np.uint32)
	print(acc(curr_preds, labels), d)

exit()

fp_fn = []
for th in np.arange(0.0, 1.0, 0.025):
	incorrect = []
	fp = []
	fn = []
	incorrect_labels = []
	incorrect_preds = []
	preds_fused_raw = preds / len(dirs)
	preds_fused = (preds_fused_raw > th).astype(np.uint32)

	for i, (x, y) in enumerate(zip(preds_fused, labels)):
		if x != y:
			incorrect.append(i)
			if y == 1:
				fn.append(i)
			else:
				fp.append(i)
			incorrect_labels.append(int(y))
			incorrect_preds.append(round(preds_fused_raw[i], 2))
	#print('Misclassified indices:', incorrect)
	#print('GT labels', incorrect_labels)
	#print('Certainty', incorrect_preds)

	fp_fn.append((len(fp), len(fn)))
	print(round(th, 2), round(acc(preds_fused, labels), 2), 'FP:', len(fp), 'FN', len(fn))
plt.plot([el[0] for el in fp_fn], [el[1] for el in fp_fn], label = 'FP / FN')
plt.xlabel("False Positives (predict: tumor, gt: healthy")
plt.ylabel("False Negatives (predict: healthy, gt: tumor")
plt.legend()
plt.savefig("fp_fn.png")
