from monai.transforms import(
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    ToNumpyd,
    Spacingd,
    Orientationd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    CropForegroundd,
    ConcatItemsd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandAffined,
    RandRotated,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandFlipd,
    EnsureTyped,
    ScaleIntensity,
    RandAdjustContrastd,
    Compose
)

#RandGaussianNoised(keys=["ct_vol", "pet_vol"], prob=0.15),
#RandGaussianSmoothd(keys=["ct_vol"], prob=0.1, sigma_x=[0.5, 1.5], sigma_y=[0.5, 1.5], sigma_z=[0.5, 1.5]),
#RandGaussianSmoothd(keys=["pet_vol"], prob=0.1, sigma_x=[0.5, 1.5], sigma_y=[0.5, 1.5], sigma_z=[0.5, 1.5]),
#RandScaleIntensityd(keys=["ct_vol", "pet_vol"], prob=0.15, factors=0.3),
#RandScaleIntensityd(keys=["ct_vol", "pet_vol"], prob=0.15, factors=[-0.45, 0.5]),
#ScaleIntensityd(keys=["ct_vol", "pet_vol"], minv=0, maxv=1),

from monai.data import set_track_meta
import numpy as np
import torch

def prepare_transforms(pixdim=(2.0, 2.0, 3.0), a_min_ct=-100, a_max_ct=250, a_min_pet=0, a_max_pet=15, spatial_size=[400, 400, 128], args=None):
    ################
    ### TRAINING ###
    ################
    if args.task == 'classification' and args.class_backbone != 'Ensemble':
        pos_freqs = {'x': 0.7, 'y': 0.65, 'z': 0.7, 'all_mips': 0.0} # only for sliding window classfication
        pos_freq = pos_freqs[args.proj_dim]

    if args.comparison == 'blackbean':
            train_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=[1.5, 1.0182, 1.0182], mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityd(keys=["ct_vol"], minv=0, maxv=1),
                    ScaleIntensityd(keys=["pet_vol"], minv=0, maxv=1),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandSpatialCropd(keys=["ct_pet_vol", "seg"], roi_size=(192, 192, 192), random_size=False),
                    RandScaleIntensityd(keys=["ct_pet_vol"], prob=1.0, factors=[-0.25, 0.25]),
                    RandAdjustContrastd(keys=["ct_pet_vol"], prob=0.3, gamma=(0.7, 1.5)),
                    RandRotated(keys=["ct_pet_vol", "seg"], range_x=np.pi / 12, range_y=np.pi / 12, range_z=np.pi / 12, prob=1.0),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["ct_pet_vol", "seg"], prob=0.5, spatial_axis=2),
                    ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
                ]
            )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["ct_vol", "pet_vol", "seg"]),
                    AddChanneld(keys=["ct_vol", "pet_vol", "seg"]),
                    Spacingd(keys=["ct_vol", "pet_vol", "seg"], pixdim=[1.5, 1.0182, 1.0182], mode=("bilinear", "bilinear", "nearest")),
                    Orientationd(keys=["ct_vol", "pet_vol", "seg"], axcodes="LAS"),
                    ScaleIntensityRanged(keys=["ct_vol"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=False),
                    ScaleIntensityd(keys=["pet_vol"], minv=0, maxv=1),
                    ConcatItemsd(keys=["ct_vol", "pet_vol"], name="ct_pet_vol", dim=0),  # concatenate pet and ct channels
                    EnsureTyped(keys=["ct_pet_vol", "seg"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
                    RandSpatialCropd(keys=["ct_pet_vol", "seg"], roi_size=(192, 192, 192), random_size=False),
                    ToTensord(keys=["seg", "ct_pet_vol", "class_label"]),
                ]
            )
            return train_transforms, val_transforms
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

    # Just generate MIP
    if args.generate_mip:
        train_transforms = Compose(
            [
                LoadImaged(keys=["pet_vol"]),
                AddChanneld(keys=["pet_vol"]),
                Spacingd(keys=["pet_vol"], pixdim=pixdim, mode=("bilinear")),
                Orientationd(keys=["pet_vol"], axcodes="LAS"),
                ScaleIntensityRanged(keys=["pet_vol"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["pet_vol"], source_key="pet_vol"),
                Resized(keys=["pet_vol"], spatial_size=spatial_size),
                ToTensord(keys=["pet_vol"]),
                EnsureTyped(keys=["pet_vol"], device=torch.device(f"cuda:{args.gpu}"), track_meta=False),
            ]
        )
    # Sliding Window Segmentation or Transference without DA
    elif args.task in ['segmentation', 'transference', 'fission', 'fission_classification', 'reconstruction'] and args.sliding_window and not args.with_DA:
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
    elif args.task in ['segmentation', 'transference'] and args.sliding_window and args.with_DA:
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
    elif (args.task in ['segmentation', 'transference', 'fission', 'fission_classification', 'reconstruction'] or args.class_backbone == 'Ensemble') and not args.with_DA:
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
    elif (args.task in ['segmentation', 'transference', 'fission', 'fission_classification', 'reconstruction'] or args.class_backbone == 'Ensemble') and args.with_DA:
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
    if (args.task in ['segmentation', 'transference', 'fission', 'fission_classification', 'reconstruction'] or args.class_backbone == 'Ensemble') and not args.sliding_window:
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
    elif (args.task in ['segmentation', 'transference', 'fission', 'fission_classification', 'reconstruction'] or args.class_backbone == 'Ensemble') and args.sliding_window:
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
