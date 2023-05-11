from monai.transforms import(
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    EnsureChannelFirstd,
    RandSpatialCropd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CropForegroundd,
    ConcatItemsd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandRotated,
    RandFlipd,
    EnsureTyped,
    Compose,
    MapTransform,
    CenterSpatialCropd,
    Transform,
    Randomizable,
    RandomizableTransform,
    MapTransform,
    RandAdjustContrastd,
    ToDeviced,
    SaveImaged,
    ShiftIntensity,
)
from monai.utils.enums import PostFix
from monai.config import DtypeLike, KeysCollection
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor

DEFAULT_POST_FIX = PostFix.meta()


import numpy as np
import torch

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 1 is Edema
            result.append(d[key] == 1)
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))

            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            d[key] = torch.stack(result, axis=0).float()
        return d

def prepare_transforms(pixdim=(2.036, 2.036, 3.0), a_min_ct=-100, a_max_ct=250, a_min_pet=0, a_max_pet=15, spatial_size=[400, 400, 128], args=None):
    ################
    ### TRAINING ###
    ################
    if args.task == 'classification' and args.class_backbone != 'Ensemble':
        pos_freqs = {'x': 0.7, 'y': 0.65, 'z': 0.7, 'all_mips': 0.0} # only for sliding window classfication
        pos_freq = pos_freqs[args.proj_dim]
        print('Positive patch sampling rate', pos_freq)

    common_tasks = ['segmentation', 'transference', 'fission', 'fission_classification', 'reconstruction', 'alt_transference']

    if args.dataset == 'ACRIN':
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=[2.0364, 2.0364, 3.0], mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=False),
                    #ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=False),
                    Resized(keys=["ct_vol", "pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
                ]
            )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=[2.0364, 2.0364, 3.0], mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityd(keys=["ct_vol"], minv=0, maxv=1),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=8.0, a_max=60000, b_min=0.0, b_max=1.0, clip=False),
                    Resized(keys=["ct_vol", "pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
                ]
            )
            return train_transforms, val_transforms

    if args.dataset == 'BraTS':
            train_transforms = Compose(
                [
                    # load 4 Nifti images and stack them together
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys="image"),
                    EnsureTyped(keys=["image", "label"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear", "nearest"),
                    ),
                    RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                ]
            )
            val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys="image"),
                    EnsureTyped(keys=["image", "label"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear", "nearest"),
                    ),
                    CenterSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144]),

                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ]
            )
            return train_transforms, val_transforms
    if args.blackbean:
        train_transforms = Compose(
            [
                LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),

                NormalizeIntensityd(keys=['pet_vol'], nonzero=True),
                ScaleIntensityRangePercentilesd(keys=['ct_vol'], lower=0.5,upper=99.5, b_min=0.0, b_max=1.0, clip=True),

                RandScaleIntensityd(keys=["ct_vol"], factors=0.25, prob=1.0),
                RandScaleIntensityd(keys=["pet_vol"], factors=0.25, prob=1.0),
                RandShiftIntensityGaussiand(keys=["ct_vol"]),
                RandShiftIntensityGaussiand(keys=["pet_vol"]),
                RandAdjustContrastd(keys=["ct_vol", "pet_vol"], prob=0.3, gamma=(1.0 / 1.5 , 1.0 / 0.7)),
                RandFlipd(keys=["ct_vol", "pet_vol", "seg"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["ct_vol", "pet_vol", "seg"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["ct_vol", "pet_vol", "seg"], prob=0.5, spatial_axis=2),
                RandRotated(keys=["ct_vol", "pet_vol", "seg"], prob=0.16, range_x=np.pi / 12.0, range_y=np.pi / 12.0, range_z=np.pi / 12.0),

                ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}")),#, track_meta=False),

                RandCropByPosNegLabeld(keys=["ct_pet_vol", "seg"], label_key="seg", spatial_size=[192, 192, 192], pos=1.0/2.0, neg=1.0/2.0, num_samples=1, allow_smaller=True),
                ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                NormalizeIntensityd(keys=['pet_vol'], nonzero=True),
                ScaleIntensityRangePercentilesd(keys=['ct_vol'], lower=0.5,upper=99.5, b_min=0.0, b_max=1.0, clip=True),
                ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
            ]
        )
        return train_transforms, val_transforms
    # Just generate MIP
    if args.generate_mip:
        train_transforms = Compose(
            [
                LoadImaged(keys=["pet_vol", "seg"]),
                AddChanneld(keys=["pet_vol", "seg"]),
                Spacingd(keys=["pet_vol", "seg"], pixdim=pixdim, mode=("bilinear")),
                Orientationd(keys=["pet_vol", "seg"], axcodes="LAS"),
                ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["pet_vol", "seg"], source_key="pet_vol"),
                Resized(keys=["pet_vol", "seg"], spatial_size=spatial_size),
                ToTensord(keys=["pet_vol", "seg"]),
                EnsureTyped(keys=["pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
            ]
        )
    # Sliding Window Segmentation or Transference without DA
    elif args.task in common_tasks and args.sliding_window and not args.with_DA:
        if args.single_mod is None: # ct_pet_vol
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=False),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=False),
                    CropForegroundd(keys=["ct_vol", "pet_vol", "seg"], source_key="pet_vol"),
                    #Resized(keys=["ct_vol", "pet_vol"], spatial_size=spatial_size),
                    #Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandCropByPosNegLabeld(keys=["ct_pet_vol", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=2.0/3.0, neg=1.0/3.0, num_samples=4, allow_smaller=True),
                    ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
                ]
            )
        elif args.single_mod == 'ct_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "seg"], pixdim=pixdim, mode=("bilinear",  "nearest")),
                    Orientationd(keys=["ct_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=False),
                    CropForegroundd(keys=["ct_vol", "seg"], source_key="ct_vol"),
                    #Resized(keys=["ct_vol"], spatial_size=spatial_size),
                    #Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    EnsureTyped(keys=["ct_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandCropByPosNegLabeld(keys=["ct_vol", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=2.0/3.0, neg=1.0/3.0, num_samples=4, allow_smaller=True),
                    ToTensord(keys=["ct_vol", "seg", "class_label"]),
                ]
            )
        elif args.single_mod == 'pet_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            train_transforms = Compose(
                [
                    LoadImaged(keys=["pet_vol", "seg"]),
                    AddChanneld(keys=["pet_vol", "seg"]),
                    Spacingd(keys=["pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=False),
                    CropForegroundd(keys=["pet_vol", "seg"], source_key="pet_vol"),
                    #Resized(keys=["pet_vol"], spatial_size=spatial_size),
                    #Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    EnsureTyped(keys=["pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandCropByPosNegLabeld(keys=["pet_vol", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=2.0/3.0, neg=1.0/3.0, num_samples=4, allow_smaller=True),
                    ToTensord(keys=["pet_vol", "seg", "class_label"]),
                ]
            )
        else:
            print(f"[ERROR] Wrong input modality!")
            exit()
    # Sliding Window Segmentation or Transference with DA
    elif args.task in common_tasks and args.sliding_window and args.with_DA:
        if args.single_mod is None: # ct_pet_vol
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=False),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=False),
                    CropForegroundd(keys=["ct_vol", "pet_vol", "seg"], source_key="pet_vol"),
                    #Resized(keys=["ct_vol", "pet_vol"], spatial_size=spatial_size),
                    #Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandCropByPosNegLabeld(keys=["ct_pet_vol", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=2.0/3.0, neg=1.0/3.0, num_samples=4, allow_smaller=True),
                    RandRotated(keys=["ct_pet_vol", "seg"], range_x=np.pi / 6, range_y=np.pi / 6, range_z=np.pi / 6, prob=0.2),
                    RandAffined(keys=["ct_pet_vol", "seg"], prob=0.2, scale_range=[-0.3, 0.4]),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=2),
                    ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
                ]
            )
        elif args.single_mod == 'ct_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "seg"], pixdim=pixdim, mode=("bilinear",  "nearest")),
                    Orientationd(keys=["ct_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=False),
                    CropForegroundd(keys=["ct_vol", "seg"], source_key="ct_vol"),
                    #Resized(keys=["ct_vol"], spatial_size=spatial_size),
                    #Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    EnsureTyped(keys=["ct_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandCropByPosNegLabeld(keys=["ct_vol", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=2.0/3.0, neg=1.0/3.0, num_samples=4, allow_smaller=True),
                    RandRotated(keys=["ct_vol", "seg"], range_x=np.pi / 6, range_y=np.pi / 6, range_z=np.pi / 6, prob=0.2),
                    RandAffined(keys=["ct_vol", "seg"], prob=0.2, scale_range=[-0.3, 0.4]),
                    RandFlipd(keys=["ct_vol", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["ct_vol", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["ct_vol", "seg"], prob=0.5, spatial_axis=2),
                    ToTensord(keys=["ct_vol", "seg", "class_label"]),
                ]
            )
        elif args.single_mod == 'pet_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            train_transforms = Compose(
                [
                    LoadImaged(keys=["pet_vol", "seg"]),
                    AddChanneld(keys=["pet_vol", "seg"]),
                    Spacingd(keys=["pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=False),
                    CropForegroundd(keys=["pet_vol", "seg"], source_key="pet_vol"),
                    #Resized(keys=["pet_vol"], spatial_size=spatial_size),
                    #Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    EnsureTyped(keys=["pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandCropByPosNegLabeld(keys=["pet_vol", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=2.0/3.0, neg=1.0/3.0, num_samples=4, allow_smaller=True),
                    RandRotated(keys=["pet_vol", "seg"], range_x=np.pi / 6, range_y=np.pi / 6, range_z=np.pi / 6, prob=0.2),
                    RandAffined(keys=["pet_vol", "seg"], prob=0.2, scale_range=[-0.3, 0.4]),
                    RandFlipd(keys=["pet_vol", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["pet_vol", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["pet_vol", "seg"], prob=0.5, spatial_axis=2),
                    ToTensord(keys=["pet_vol", "seg", "class_label"]),
                ]
            )
        else:
            print(f"[ERROR] Wrong input modality!")
            exit()
    # Normal Segmentation or Transference (without DA)
    elif (args.task in common_tasks or args.class_backbone == 'Ensemble') and not args.with_DA:
        if args.single_mod is None: # input_mod == ct_pet_vol
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "pet_vol", "seg"], source_key="pet_vol"),
                    Resized(keys=["ct_vol", "pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    ToTensord(keys=["ct_vol", "pet_vol", "seg", "ct_pet_vol", "class_label"]),
                    EnsureTyped(keys=["seg", "ct_pet_vol", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        elif args.single_mod == 'ct_vol':
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "seg"], source_key="ct_vol"),
                    Resized(keys=["ct_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ToTensord(keys=["ct_vol", "seg", "class_label"]),
                    EnsureTyped(keys=["ct_vol", "seg", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        elif args.single_mod == 'pet_vol':
            train_transforms = Compose(
                [
                    LoadImaged(keys=["pet_vol", "seg"]),
                    AddChanneld(keys=["pet_vol", "seg"]),
                    Spacingd(keys=["pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["pet_vol", "seg"], source_key="pet_vol"),
                    Resized(keys=["pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ToTensord(keys=["pet_vol", "seg", "class_label"]),
                    EnsureTyped(keys=["pet_vol", "seg", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        else:
            print(f"[ERROR] Wrong input modality!")
            exit()
    # Normal Segmentation or Transference (with DA)
    elif (args.task in common_tasks or args.class_backbone == 'Ensemble') and args.with_DA:
        print("Using transforms WITH data augmentation.")
        if args.single_mod is None: # input_mod == ct_pet_vol
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "pet_vol", "seg"], source_key="pet_vol"),
                    Resized(keys=["ct_vol", "pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandRotated(keys=["ct_pet_vol", "seg"], range_x=np.pi / 6, range_y=np.pi / 6, range_z=np.pi / 6, prob=0.2),
                    RandAffined(keys=["ct_pet_vol", "seg"], prob=0.2, scale_range=[-0.3, 0.4]),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=2),
                    #ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
                ]
            )
        elif args.single_mod == 'ct_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "seg"], source_key="ct_vol"),
                    Resized(keys=["ct_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    EnsureTyped(keys=["ct_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandRotated(keys=["ct_vol", "seg"], range_x=np.pi / 6, range_y=np.pi / 6, range_z=np.pi / 6, prob=0.2),
                    RandAffined(keys=["ct_vol", "seg"], prob=0.2, scale_range=[-0.3, 0.4]),
                    RandFlipd(keys=["ct_vol", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["ct_vol", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["ct_vol", "seg"], prob=0.5, spatial_axis=2),
                    ToTensord(keys=["ct_vol", "seg", "class_label"]),
                ]
            )
        elif args.single_mod == 'pet_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            train_transforms = Compose(
                [
                    LoadImaged(keys=["pet_vol", "seg"]),
                    AddChanneld(keys=["pet_vol", "seg"]),
                    Spacingd(keys=["pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["pet_vol", "seg"], source_key="pet_vol"),
                    Resized(keys=["pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    EnsureTyped(keys=["pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandRotated(keys=["pet_vol", "seg"], range_x=np.pi / 6, range_y=np.pi / 6, range_z=np.pi / 6, prob=0.2),
                    RandAffined(keys=["pet_vol", "seg"], prob=0.2, scale_range=[-0.3, 0.4]),
                    RandFlipd(keys=["pet_vol", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["pet_vol", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["pet_vol", "seg"], prob=0.5, spatial_axis=2),
                    ToTensord(keys=["pet_vol", "seg", "class_label"]),
                ]
            )
        else:
            print(f"[ERROR] Wrong input modality!")
            exit()
    # Normal Classification
    elif args.task == 'classification' and not args.sliding_window:
        train_transforms = Compose(
            [
                LoadImaged(keys=["mip_x", "mip_y", "mip_z"]),
                AddChanneld(keys=["mip_x", "mip_y", "mip_z"]),
                RandAffined(keys=["mip_x", "mip_y", "mip_z"], prob=0.7, rotate_range=np.pi / 18, translate_range = 0.0625 * 400, scale_range=0.1),
                Resized(keys=["mip_x", "mip_y", "mip_z"], spatial_size=spatial_size[:2]),
                ConcatItemsd(keys=["mip_x", "mip_y", "mip_z"], name="all_mips", dim=0),  # concatenate all MIPs
                ToTensord(keys=["mip_x", "mip_y", "mip_z", "all_mips", "class_label"]),
                EnsureTyped(keys=["mip_x", "mip_y", "mip_z", "all_mips", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
            ]
        )
    # Sliding Window Classification
    elif args.task == 'classification' and args.sliding_window:
        train_transforms = Compose(
            [
                LoadImaged(keys=["mip_x", "mip_y", "mip_z", 'mip_seg_x', 'mip_seg_y', 'mip_seg_z']),
                AddChanneld(keys=["mip_x", "mip_y", "mip_z", "mip_seg_x", "mip_seg_y", "mip_seg_z"]),
                RandCropByPosNegLabeld(keys=["mip_x", "mip_y", "mip_z", "mip_seg_x", "mip_seg_y", "mip_seg_z"], label_key=f"mip_seg_{args.proj_dim}", spatial_size=[100, 100], pos=pos_freq, neg=(1 - pos_freq), num_samples=4, allow_smaller=True),
                ConcatItemsd(keys=["mip_x", "mip_y", "mip_z"], name="all_mips", dim=0),  # concatenate all MIPs
                ToTensord(keys=["mip_x", "mip_y", "mip_z", "all_mips", "class_label", "mip_seg_x", "mip_seg_y", "mip_seg_z"]),
                EnsureTyped(keys=["mip_x", "mip_y", "mip_z", "all_mips", "class_label", "mip_seg_x", "mip_seg_y", "mip_seg_z"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
            ]
        )
    else:
        print("Unsupported train transforms...")
        exit()
    ##################
    ### VALIDATION ###
    ##################
    # Normal Segmentation or Transference
    if (args.task in common_tasks or args.class_backbone == 'Ensemble') and not args.sliding_window:
        if args.single_mod is None: # input_mod == ct_pet_vol
            val_transforms= Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "pet_vol", "seg"], source_key="pet_vol"),
                    Resized(keys=["ct_vol", "pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    ToTensord(keys=["ct_vol", "pet_vol", "seg", "ct_pet_vol", "class_label"]),
                    EnsureTyped(keys=["seg", "ct_pet_vol", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        elif args.single_mod == 'ct_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            val_transforms= Compose(
                [
                    LoadImaged(keys=["ct_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "seg"], source_key="ct_vol"),
                    Resized(keys=["ct_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ToTensord(keys=["ct_vol", "seg", "class_label"]),
                    EnsureTyped(keys=["ct_vol", "seg", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        elif args.single_mod == 'pet_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            val_transforms= Compose(
                [
                    LoadImaged(keys=["pet_vol", "seg"]),
                    AddChanneld(keys=["pet_vol", "seg"]),
                    Spacingd(keys=["pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["pet_vol", "seg"], source_key="pet_vol"),
                    Resized(keys=["pet_vol"], spatial_size=spatial_size),
                    Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
                    ToTensord(keys=["pet_vol", "seg", "class_label"]),
                    EnsureTyped(keys=["pet_vol", "seg", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),

                ]
            )
        else:
            print(f"[ERROR] Wrong input modality!")
            exit()
    elif (args.task in common_tasks or args.class_backbone == 'Ensemble') and args.sliding_window:
        if args.single_mod is None: # input_mod == ct_pet_vol
            val_transforms= Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "pet_vol", "seg"], source_key="pet_vol"),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    ToTensord(keys=["ct_vol", "pet_vol", "seg", "ct_pet_vol", "class_label"]),
                    EnsureTyped(keys=["seg", "ct_pet_vol", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        elif args.single_mod == 'ct_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()

            val_transforms= Compose(
                [
                    LoadImaged(keys=["ct_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["ct_vol", "seg"], source_key="ct_vol"),
                    ToTensord(keys=["ct_vol", "seg", "class_label"]),
                    EnsureTyped(keys=["ct_vol", "seg", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        elif args.single_mod == 'pet_vol':
            if args.task == 'transference':
                print(f'[ERROR] Cannot do transference with only one modality {args.single_mod}!')
                exit()
            val_transforms= Compose(
                [
                    LoadImaged(keys=["pet_vol", "seg"]),
                    AddChanneld(keys=["pet_vol", "seg"]),
                    Spacingd(keys=["pet_vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
                    Orientationd(keys=["pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["pet_vol", "seg"], source_key="pet_vol"),
                    ToTensord(keys=["pet_vol", "seg", "class_label"]),
                    EnsureTyped(keys=["pet_vol", "seg", "class_label"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                ]
            )
        else:
            print(f"[ERROR] Wrong input modality!")
            exit()

    # Normal Classification
    elif args.task == 'classification':
        val_transforms = Compose(
            [
                LoadImaged(keys=["mip_x", "mip_y", "mip_z", 'mip_seg_x', 'mip_seg_y', 'mip_seg_z']),
                AddChanneld(keys=["mip_x", "mip_y", "mip_z", "mip_seg_x", "mip_seg_y", "mip_seg_z"]),
                Resized(keys=["mip_x", "mip_y", "mip_z", "mip_seg_x", "mip_seg_y", "mip_seg_z"], spatial_size=spatial_size[:2]),
                ConcatItemsd(keys=["mip_x", "mip_y", "mip_z"], name="all_mips", dim=0),  # concatenate all MIPs
                ToTensord(keys=["mip_x", "mip_y", "mip_z", "all_mips", "class_label", "mip_seg_x", "mip_seg_y", "mip_seg_z"]),
                EnsureTyped(keys=["mip_x", "mip_y", "mip_z", "all_mips", "class_label", "mip_seg_x", "mip_seg_y", "mip_seg_z"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
            ]
        )
    else:
        print("Unsupported val transforms...")
        exit()

    return train_transforms, val_transforms

class RandShiftIntensityGaussian(Randomizable, Transform):
    """Randomly shift intensity with randomly picked offset.
    """

    def __init__(self, prob=1.0):
        self.prob = prob
        self._do_transform = False

    def randomize(self):
        self._offset = self.R.normal(loc=0, scale=0.1) # offset ~ N(0, 0.1)
        self._do_transform = self.R.random() < self.prob


    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        shifter = ShiftIntensity(self._offset)
        return shifter(img)

class RandShiftIntensityGaussiand(RandomizableTransform, MapTransform):
    backend = RandShiftIntensityGaussian.backend

    def __init__(
            self,
            keys: KeysCollection,
            meta_key_postfix: str = DEFAULT_POST_FIX,
            prob: float = 1.0,
            allow_missing_keys: bool = False,
        ) -> None:

            MapTransform.__init__(self, keys, allow_missing_keys)
            RandomizableTransform.__init__(self, prob)

            self.shifter = RandShiftIntensityGaussian(prob=1.0)


    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
        ) -> "RandShiftIntensityGaussiand":
            super().set_random_state(seed, state)
            self.shifter.set_random_state(seed, state)
            return self


    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
            d = dict(data)
            self.randomize(None)
            if not self._do_transform:
                for key in self.key_iterator(d):
                    d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
                return d

            # all the keys share the same random shift factor
            self.shifter.randomize()
            for key in self.key_iterator(
                d
            ):
                d[key] = self.shifter(d[key])
            return d
