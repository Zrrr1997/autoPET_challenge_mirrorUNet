import numpy as np
from matplotlib import pyplot as plt

outputs = [line.rstrip() for line in open('./data/aux/output_evals_both', 'r')  if 'resnet50' in line and 'mip_z' not in line]

labels = [el.replace('outputs.npy', 'labels.npy') for el in outputs]


accs = []
hyp_theta = False
if hyp_theta:
	for theta in np.arange(0.0, 1.0, 0.01):
		outs = np.array([np.load(el) for el in outputs])
		n_els = outs.shape[0]
		th = n_els * theta
		outs_raw = np.sum(outs, axis=0) / n_els
		outs = (np.sum(outs, axis=0) > th) * 1.0


		gts = np.array([np.load(el) for el in labels])


		errors = (outs != gts[0]) * 1.0

		accs.append(np.sum(1 - errors) / len(errors))
	plt.plot(accs)
	locs, labels = plt.xticks()


	plt.xticks(locs, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
	plt.xlabel("sigmoid thershold")
	plt.ylabel("Accuracy")

	plt.savefig('ROC.png')

	max_el = max(accs)
	print('Best threshold', accs.index(max_el) * 0.01)
	print('Best accuracy', round(max(accs), 2))
	exit()

plot = True

if plot:
	outs = np.array([np.load(el) for el in outputs])
	n_els = outs.shape[0]
	th = n_els * 0.4
	outs_raw = np.sum(outs, axis=0) / n_els
	outs = (np.sum(outs, axis=0) > th) * 1.0


	gts = np.array([np.load(el) for el in labels])


	errors = (outs != gts[0]) * 1.0

	accs.append(np.sum(1 - errors) / len(errors))
	err_indices = np.argwhere(errors == 1).flatten()

	plt.hist(outs_raw, bins=20, label = 'All Samples')
	#plt.savefig('average_certainty.png')

	err_outs_raw = []

	for ind in err_indices:
		print('Index:', ind, 'GT:', gts[0][ind], round(outs_raw[ind], 2))
		err_outs_raw.append(outs_raw[ind])
	plt.hist(err_outs_raw, bins=20, label = 'Misclassified Samples')
	plt.xlabel("Sigmoid Output")
	plt.ylabel("# Samples")
	plt.legend()
	plt.savefig('average_certainty_distribution.png')
