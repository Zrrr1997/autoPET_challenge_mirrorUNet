import os
import cv2
import torch
import torch.nn.functional as nnf

import nibabel as nib
import numpy as np

from tqdm import tqdm

from monai.transforms import AddChannel
from utils.transforms import prepare_transforms


def generate_mip(fns, d_str, args):
    print(f"Generating MIPs for {d_str} set...")

    train_transforms, _ = prepare_transforms(args=args)

    # Generate MIP projections
    add_channel = AddChannel()

    cls_labels = []
    for i, fn in tqdm(enumerate(fns)):
        cls_labels.append(fn['class_label'])



        # Assume MIP is done on PET data...
        pet_vol = train_transforms(fn)['pet_vol'][0]
        assert pet_vol.shape == (400, 400, 128)

        mip_x = add_channel(add_channel(torch.max(pet_vol, dim=0)[0]))
        mip_y = add_channel(add_channel(torch.max(pet_vol, dim=1)[0]))
        mip_z = add_channel(add_channel(torch.max(pet_vol, dim=2)[0]))

        mip_x = nnf.interpolate(mip_x, size=(400, 400), mode='bicubic', align_corners=False).numpy()[0][0]
        mip_y = nnf.interpolate(mip_y, size=(400, 400), mode='bicubic', align_corners=False).numpy()[0][0]
        mip_z = nnf.interpolate(mip_z, size=(400, 400), mode='bicubic', align_corners=False).numpy()[0][0]

        np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/mip_x/{i}.npy', mip_x)
        np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/mip_y/{i}.npy', mip_y)
        np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/mip_z/{i}.npy', mip_z)


    np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/class_labels.npy', np.array(cls_labels))
