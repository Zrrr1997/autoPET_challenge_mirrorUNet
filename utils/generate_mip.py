import os
import cv2
import torch
import torch.nn.functional as nnf

import nibabel as nib
import numpy as np

from tqdm import tqdm

from monai.transforms import AddChannel
from utils.transforms import prepare_transforms

# Generate Maximum Intensity Projections
def generate_mip(fns, d_str, args):
    print(f"Generating MIPs for {d_str} set...")

    # Never generate MIPs with only positive samples
    assert args.with_negatives

    train_transforms, _ = prepare_transforms(args=args)

    # Generate MIP projections
    add_channel = AddChannel()

    cls_labels = []
    for i, fn in tqdm(enumerate(fns)):
        cls_labels.append(fn['class_label'])


        # Assume MIP is done on PET data...
        pet_vol = train_transforms(fn)['pet_vol'][0]
        seg_vol = train_transforms(fn)['pet_vol'][0]

        assert pet_vol.shape == (400, 400, 128) and pet_vol.shape == seg_vol.shape

        mip_x = add_channel(add_channel(torch.max(pet_vol, dim=0)[0]))
        mip_y = add_channel(add_channel(torch.max(pet_vol, dim=1)[0]))
        mip_z = add_channel(add_channel(torch.max(pet_vol, dim=2)[0]))

        seg_x = add_channel(add_channel(torch.max(seg_vol, dim=0)[0]))
        seg_y = add_channel(add_channel(torch.max(seg_vol, dim=1)[0]))
        seg_z = add_channel(add_channel(torch.max(seg_vol, dim=2)[0]))


        mip_x = nnf.interpolate(mip_x, size=(400, 400), mode='bicubic', align_corners=False).cpu().detach().numpy()[0][0]
        mip_y = nnf.interpolate(mip_y, size=(400, 400), mode='bicubic', align_corners=False).cpu().detach().numpy()[0][0]
        mip_z = nnf.interpolate(mip_z, size=(400, 400), mode='bicubic', align_corners=False).cpu().detach().numpy()[0][0]

        seg_x = nnf.interpolate(seg_x, size=(400, 400), mode='bicubic', align_corners=False).cpu().detach().numpy()[0][0]
        seg_y = nnf.interpolate(seg_y, size=(400, 400), mode='bicubic', align_corners=False).cpu().detach().numpy()[0][0]
        seg_z = nnf.interpolate(seg_z, size=(400, 400), mode='bicubic', align_corners=False).cpu().detach().numpy()[0][0]

        seg_x = (seg_x > 0) * 1
        seg_y = (seg_y > 0) * 1
        seg_z = (seg_z > 0) * 1


        print('saving', f'./data/MIP/fold_{args.fold}/{d_str}_data/mip_x/{str(i).zfill(4)}.npy')
        np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/mip_x/{str(i).zfill(4)}.npy', mip_x)
        np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/mip_y/{str(i).zfill(4)}.npy', mip_y)
        np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/mip_z/{str(i).zfill(4)}.npy', mip_z)


    np.save(f'./data/MIP/fold_{args.fold}/{d_str}_data/class_labels.npy', np.array(cls_labels))
