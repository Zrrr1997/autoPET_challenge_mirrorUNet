# MONAI
from monai.utils import first
from monai.losses import DiceLoss, DiceCELoss
from monai.handlers import MeanDice, MeanSquaredError
from monai.transforms import (
    AsDiscrete,
    Identity,
    Lambda
)

# Ignite
from ignite.metrics import Accuracy
import torch.nn.functional as nnf

import cv2
import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

from loss.dice_ce_rec import DiceCE_Rec_Loss # Transference, Fission
from loss.dice_ce_rec_class import DiceCE_Rec_Class_Loss # Fission + Classification
from loss.dice_ce_brats import DiceCE_BraTS_Loss

# Visualization
from torchinfo import summary
from torchviz import make_dot


def check_args(args):
    if args.sliding_window:
        assert args.batch_size == 1 # Sliding window can only work with batch_size 1
    if args.task == 'classification':
        assert args.batch_size >= 4 and args.proj_dim is not None
    if args.depth != 1:
        assert args.level == 3 # Only share around the bottleneck with higher depth
    if 'fission' in args.task:
        assert args.batch_size <= 2 # Model is too large for larger batch sizes

def prepare_out_channels(args):
    out_channels = 2 if args.task in ['segmentation',
                                    'segmentation_classification',
                                    'transference',
                                    'fission',
                                    'fission_classification',
                                    'alt_transference'] or args.early_fusion else 1

    print('Number of output channels', out_channels)
    return out_channels


def prepare_input_mod(args):
    if args.dataset == 'BraTS':
        return 'image'
    input_mod = "ct_pet_vol" if args.single_mod is None or args.early_fusion else args.single_mod
    input_mod = f'mip_{args.proj_dim}' if args.proj_dim is not None and args.proj_dim != "all_mips" else input_mod
    input_mod = "all_mips" if args.proj_dim == 'all_mips' else input_mod
    assert input_mod in ['ct_pet_vol', 'ct_vol', 'pet_vol', 'mip_x', 'mip_y', 'mip_y', 'mip_z']
    print('Input modality:', input_mod)
    return input_mod


def check_data_shape(train_loader, input_mod, args):
    check_data = first(train_loader)

    print('Input shape check:', check_data[input_mod].shape)
    if args.task == 'segmentation':
        if args.dataset == 'BraTS':
            print('Segmentation mask shape check:', check_data["label"].shape)
        else:
            print('Segmentation mask shape check:', check_data["seg"].shape)
            print('Class Label', check_data["class_label"])


        mip_x = torch.max(check_data[input_mod][0][0], axis=0)[0]
        print('min_val:', np.min(mip_x), 'max_val:', np.max(mip_x))

    elif args.task == 'classification':
        mip_x = check_data[input_mod][0][0].cpu().detach().numpy()
        print('min_val_mip:', np.min(mip_x), 'max_val_mip:', np.max(mip_x))
        print('mip_x.shape', mip_x.shape)


def prepare_loss(args):
    if args.dataset == 'BraTS':
        loss = DiceCE_BraTS_Loss(to_onehot_y=True, softmax=True, include_background=args.include_background, batch=True, args=args)
    elif args.task in ['segmentation', 'alt_transference']:
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
    elif args.task in ['transference', 'fission']:
        loss = DiceCE_Rec_Loss(to_onehot_y=True, softmax=True, include_background=args.include_background, batch=True, lambda_rec=args.lambda_rec, lambda_ce=args.lambda_seg, lambda_dice=args.lambda_seg, args=args)
        print('Using DiceCE loss with reconstruction.')
    elif args.task == 'fission_classification': # TODO
        loss = DiceCE_Rec_Class_Loss(to_onehot_y=True, softmax=True, include_background=args.include_background, batch=True, lambda_rec=args.lambda_rec, lambda_ce=args.lambda_seg, lambda_dice=args.lambda_seg)
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

def prepare_val_metrics(args):
    if args.task in ['segmentation', 'segmentation_classification', 'transference', 'fission', 'fission_classification', 'alt_transference']:
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

def save_nifti_img(name, im):
    print(name, im.shape)
    affine = np.eye(4)
    affine[0][0] = -1
    ni_img = nib.Nifti1Image(im, affine=affine)
    ni_img.header.get_xyzt_units()
    ni_img.to_filename(f'{name}.nii.gz')

# Assume x is in shape CHWD
def transference_post_pred(x):
    num_classes=2
    discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    #return discrete(x) # only for args.save_nifti == True
    return discrete(x[1:,:])


def transference_post_label(x):
    num_classes=2
    discrete = AsDiscrete(to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    return discrete(x[1:,:])

def fission_post_pred(x):
    num_classes=2
    discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    if x.shape[0] > 2:
        return discrete(x[2:4,:])
    else:
        return discrete(x)

def brats_post_pred(x):
    num_classes=2
    discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    return discrete(x[:2] + x[4:]) # Late fusion of Core and Edema predictions
def brats_post_label(x):
    num_classes = 2
    discrete = AsDiscrete(to_onehot=num_classes)
    return discrete(x[1:2]) # Whole tumors

def brats_post_pred_core(x):
    num_classes=2
    discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    return discrete(x[:2]) # Core prediction
def brats_post_label_core(x):
    num_classes = 2
    discrete = AsDiscrete(to_onehot=num_classes)
    return discrete(x[:1]) # Core tumor

def brats_post_pred_edema(x):
    num_classes=2
    discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    return discrete(x[4:]) # Edema predictions
def brats_post_label_edema(x):
    num_classes = 2
    discrete = AsDiscrete(to_onehot=num_classes)
    return discrete(x[2:]) # Edemas of tumors


def fission_post_label(x):
    num_classes=2
    discrete = AsDiscrete(to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    if x.shape[0] > 2:
        return discrete(x[2:4,:])
    else:
        return discrete(x)

def fission_cls_post_label(x):
    num_classes=2
    discrete = AsDiscrete(to_onehot=num_classes)
    #return torch.cat([x[:1,:], discrete(x[1:,:])], dim=0)
    return discrete(x[2:3,:])


def prepare_post_fns(args):
    num_classes = 2
    if args.dataset == 'BraTS':
        post_pred = Lambda(func=lambda x: brats_post_pred(x))
        post_label = Lambda(func=lambda x: brats_post_label(x))

    elif args.task in ['segmentation', 'alt_transference']:
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
    elif args.task in ['fission', 'fission_classification']:
        post_pred = Lambda(func=lambda x: fission_post_pred(x))
        if args.task == 'fission_classification':
            post_label = Lambda(func=lambda x: fission_cls_post_label(x))
        else:
            post_label = Lambda(func=lambda x: fission_post_label(x))

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

def save_network_graph_plot(net, data, args):
    yhat = net(data)
    make_dot(yhat, params=dict(list(net.named_parameters()))).render(f"mirrorUNet_graph_level_{args.level}_depth_{args.depth}", format="png")
