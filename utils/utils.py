# MONAI
from monai.utils import first
from monai.losses import DiceLoss, DiceCELoss
from monai.handlers import MeanDice, MeanSquaredError
from monai.transforms import (
    AsDiscrete,
    Identity,
    Compose,
    ConcatItemsd,
    SplitChannel,
    Lambda
)

# Ignite
from ignite.metrics import Accuracy
import torch.nn.functional as nnf

import cv2
import os
import torch
import numpy as np
from tqdm import tqdm

from loss.dice_ce_rec import DiceCE_Rec_Loss



def prepare_out_channels(args):
    out_channels = 2 if args.task in ['segmentation', 'segmentation_classification', 'transference'] or args.early_fusion else 1

    print('Number of output channels', out_channels)
    return out_channels

def prepare_input_mod(args):
    input_mod = "ct_pet_vol" if args.single_mod is None or args.early_fusion else args.single_mod
    input_mod = f'mip_{args.proj_dim}' if args.proj_dim is not None and args.proj_dim != "all_mips" else input_mod
    input_mod = "all_mips" if args.proj_dim == 'all_mips' else input_mod
    print('Input modality:', input_mod)
    return input_mod

def check_data_shape(train_loader, input_mod, args):
    check_data = first(train_loader)

    print('Input shape check:', check_data[input_mod].shape)
    if args.task == 'segmentation':
        print('Segmentation mask shape check:', check_data["seg"].shape)

        mip_x = torch.max(check_data[input_mod][0][0], axis=0)[0]
        mip_x = nnf.interpolate(mip_x.unsqueeze(dim=0).unsqueeze(dim=0), size=(400, 400), mode='bicubic', align_corners=False).cpu().detach().numpy()[0][0]
        print('min_val_mip:', np.min(mip_x), 'max_val_mip:', np.max(mip_x))
        print('mip_x.shape', mip_x.shape)

    elif args.task == 'classification':
        mip_x = check_data[input_mod][0][0].cpu().detach().numpy()
        print('min_val_mip:', np.min(mip_x), 'max_val_mip:', np.max(mip_x))
        print('mip_x.shape', mip_x.shape)



    print('Label', check_data["class_label"])

def prepare_loss(args):
    if args.task == 'segmentation' or args.task == 'segmentation_classification':
        if args.loss == 'DiceCE':
            loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=args.include_background, batch=True)
        elif args.loss == 'Dice':
            loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=args.include_background, batch=True)
        else:
            raise ValueError(f"Loss {args.loss} is not supported!")
        print(f'Using {args.loss} loss for segmentation with include_background={args.include_background}')

    elif args.task == 'reconstruction':
        loss = torch.nn.MSELoss()
    elif args.task == 'classification':
        loss = torch.nn.BCELoss()
    elif args.task == 'transference':
        loss = DiceCE_Rec_Loss(to_onehot_y=True, softmax=True, include_background=args.include_background, batch=True)
        print('Using DiceCE loss with reconstruction.')
    else:
        raise ValueError(f"Task {args.task} is not supported!")
    return loss

def prepare_attention(args):
    if args.mask_attention and args.task == 'segmentation':
        attention = torch.tensor(np.load('data/normalized_sums.npy').astype(np.float32)).unsqueeze(dim=0)
        attention = torch.cat(args.batch_size * [attention])
    elif args.mask_attention and args.task == 'classification':
        if args.proj_dim is None or args.proj_dim == 'all_mips':
            print(f'Attention for {args.proj_dim} is not supported!')
        attention = torch.tensor(np.max(cv2.imread(f'./data/MIP/train_data/mip_{args.proj_dim}_sum.png'), axis=2)).unsqueeze(dim=0).unsqueeze(dim=0)
    return attention

def class_label(ct_path, neg_paths):
    if ct_path[:-12] in neg_paths: # len(CTres.nii.gz) == 12
        return 0.0
    return 1.0

def prepare_val_metrics(args):
    if args.task in ['segmentation', 'segmentation_classification', 'transference']:
        metric_name = "Mean_Dice_F1"
        metric_name_2 = "Mean_Dice"
        val_metrics = {metric_name: MeanDice(include_background=False), metric_name_2: MeanDice()}
    elif args.task == 'reconstruction':
        metric_name = "MSE"
        val_metrics = {metric_name: MeanSquaredError()}
    elif args.task == 'classification':
        metric_name = "Accuracy"
        val_metrics = {metric_name: Accuracy()}
    else:
        print("[ERROR] Validations metrics for such a task not found.")
        exit()
    return val_metrics

# Assume x is in shape CHWD
def transference_post_pred(x):
    num_classes=2
    discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    return discrete(x[1:,:])


def transference_post_label(x):
    num_classes=2
    discrete = AsDiscrete(to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    return discrete(x[1:,:])


def prepare_post_fns(args):
    num_classes = 2
    if args.task == 'segmentation':
        post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        post_label = AsDiscrete(to_onehot=num_classes)
    elif args.task == 'reconstruction':
        post_pred = Identity()
        post_label = Identity()
    elif args.task == 'classification':
        post_pred = AsDiscrete(threshold=0.5)
        post_label = Identity()
    elif args.task == 'transference':
        post_pred = Lambda(func=lambda x: transference_post_pred(x))
        post_label = Lambda(func=lambda x: transference_post_label(x))
    else:
        print('[ERROR] post_pred and post_label cannot be created for this task.')
        exit()

    return post_pred, post_label

def generate_pngs(data_dir):
    png_dir = os.path.join(data_dir, 'pngs')
    if not os.path.exists(png_dir):
        os.mkdir(png_dir)
        os.mkdir(os.path.join(png_dir, 'mip_x'))
        os.mkdir(os.path.join(png_dir, 'mip_y'))
        os.mkdir(os.path.join(png_dir, 'mip_z'))
    for axis in ['x', 'y', 'z']:
        img_fns = sorted([os.path.join(data_dir, f'mip_{axis}',  el) for el in sorted(os.listdir(os.path.join(data_dir, f'mip_{axis}')))])
        for img_fn in tqdm(img_fns):
            img = np.load(img_fn)
            img_png = np.expand_dims(img, axis=2) * 255
            img_png_fn = os.path.join(data_dir, f'pngs/mip_{axis}', img_fn.split('/')[-1].replace('npy', 'png'))
            cv2.imwrite(img_png_fn, img_png)