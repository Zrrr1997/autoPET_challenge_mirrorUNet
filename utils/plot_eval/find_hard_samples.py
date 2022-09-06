import numpy as np
from matplotlib import pyplot as plt

outputs = [line.rstrip() for line in open('./data/aux/output_evals_debrain', 'r')  if 'mip_y' in line or 'mip_x' in line]

labels = [el.replace('outputs.npy', 'labels.npy') for el in outputs]

n_samples = 202
error_freq = np.zeros(n_samples)

accs = []

plot = True

for i, single_outputs in enumerate(outputs):
	if plot:
		outs = np.array(np.load(single_outputs))

		th = 0.5
		outs = (outs > th) * 1.0


		gts = np.array(np.load(labels[i]))

		errors = (outs != gts) * 1.0

		#print("Accuracy:", round(np.sum(1 - errors) / len(errors), 3))

		err_indices = np.argwhere(errors == 1).flatten()


		err_outs_raw = []


		for ind in err_indices:
			error_freq[ind] += 1



top_5_worst = np.argpartition(error_freq, -10)[-10:]
gt_labels_worst = [gts[el] for el in top_5_worst]
freq_worst = [error_freq[el] for el in top_5_worst]
#print('Worst samples:', top_5_worst)
#print('GT labels:', gt_labels_worst)
worst_with_freq = list(zip(top_5_worst, [round(el / len(outputs), 2) * 100 for el in freq_worst], gt_labels_worst))
worst_with_freq.sort(key = lambda tup: tup[1], reverse=True)

print("\n(index, freqeuncy of error [%], GT-label)")
print(worst_with_freq, '\n')


#print('Frequency [%]:', [round(el / len(outputs), 2) * 100 for el in freq_worst])
