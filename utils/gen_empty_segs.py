import numpy as np
import os
import cv2

empty_seg = np.zeros((400, 400, 1))
for i in range(1, 5):
    for axis in ['x', 'y', 'z']:
        mip_train = f'./data/MIP/fold_{i}/train_data/mip_{axis}'
        mip_val = f'./data/MIP/fold_{i}/val_data/mip_{axis}'
        n_train = len(os.listdir(mip_train))
        n_val = len(os.listdir(mip_val))
        for t in range(n_train):
            cv2.imwrite(f'./data/MIP/fold_{i}/train_data/pngs/seg/mip_{axis}/{t}.png', empty_seg)
        for t in range(n_val):
            cv2.imwrite(f'./data/MIP/fold_{i}/val_data/pngs/seg/mip_{axis}/{t}.png', empty_seg)
