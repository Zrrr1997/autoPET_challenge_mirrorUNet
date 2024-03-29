# Segmentation Networks
from models.mirror_UNet_variable import Mirror_UNet
from models.UNet import UNet

# Classification Networks
from models.classification.resnet import ResNet
from models.classification.efficient_net import EfficientNet
from models.classification.ensemble import Ensemble
from models.classification.coatnet import *

# Utils
from monai.networks.layers import Norm
import torch.nn as nn
import os

def prepare_model(device=None, out_channels=None, args=None, second=False):
    net_2 = None
    if args.brats_unet:
        net = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=6, # softmax output (1 channel per class, i.e. Fg/Bg), 1 channel only for reconstruction (SSL pre-training)
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            device=device
        ).to(device)
        print("Using BraTS UNet")
    elif args.blackbean:
        net = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2, # softmax output (1 channel per class, i.e. Fg/Bg), 1 channel only for reconstruction (SSL pre-training)
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            device=device
        ).to(device)
        print("Using Blackbean UNet")
    # Segmentation / Reconstruction
    elif (args.single_mod is not None or args.early_fusion) and args.task != 'classification' and args.class_backbone != 'Ensemble':
        in_channels = 2 if args.early_fusion and args.load_weights_second_model is None else 1
        in_channels = in_channels + 1 if args.mask_attention else in_channels
        net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels, # softmax output (1 channel per class, i.e. Fg/Bg), 1 channel only for reconstruction (SSL pre-training)
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            device=device
        ).to(device)
        if args.load_weights_second_model:
            net_2 = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels, # softmax output (1 channel per class, i.e. Fg/Bg), 1 channel only for reconstruction (SSL pre-training)
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
                device=device
            ).to(device)
        print(f"Using UNet with early fusion == {args.early_fusion}")
    elif args.task != 'classification' and args.class_backbone != 'Ensemble': # Reconstruction, Segmentation or Transference
        in_channels = 2 if args.mask_attention else 1
        net = Mirror_UNet(
            spatial_dims=3,
            in_channels=1, # must be left at 1, this refers to the #c of each individual branch (PET or CT)
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            task=args.task,
            args=args,
        ).to(device)
        print("Using Mirror-UNet")
    else: # classification
        in_channels = 2 if args.mask_attention else 3 # Attention Mask not implemented for 3-channel MIPs (xyz)
        if not args.mask_attention and args.proj_dim != 'all_mips':
            in_channels = 1
        print("Number of input channels:", in_channels)

        if args.class_backbone == 'ResNet':
            net = ResNet(resnet_v=args.resnet_version, in_channels=in_channels, args=args).to(device)
            if args.resnet_version not in ['resnet18', 'resnet50', 'resnet101']:
                print(args.resnet_version, ' ResNet version not supported!')
                exit()
            print(f"Using ResNet {args.resnet_version} for classification")
        elif args.class_backbone == 'EfficientNet':
            net = EfficientNet(net_v=args.efficientnet_version, in_channels=in_channels, args=args).to(device)
        elif args.class_backbone == 'CoAtNet':
            image_size = (224, 224)
            if args.coatnet_version == '0':
                net = coatnet_0(image_size=image_size, in_channels=in_channels, num_classes=1).to(device)
            elif args.coatnet_version == '1':
                net = coatnet_1(image_size=image_size, in_channels=in_channels, num_classes=1).to(device)
            elif args.coatnet_version == '2':
                net = coatnet_2(image_size=image_size, in_channels=in_channels, num_classes=1).to(device)
            elif args.coatnet_version == '3':
                net = coatnet_3(image_size=image_size, in_channels=in_channels, num_classes=1).to(device)
            elif args.coatnet_version == '4':
                net = coatnet_4(image_size=image_size, in_channels=in_channels, num_classes=1).to(device)
            else:
                print(f"[ERROR] Version {args.coatnet_version} of CoAtNet does not exist!")
                exit()
        elif args.class_backbone == 'Ensemble':
            net = Ensemble(resnet18_x_fn = f'./checkpoints/classification/resnet18/mip_x_pet_resnet18/fold_{args.fold}',
                    resnet18_y_fn = f'./checkpoints/classification/resnet18/mip_y_pet_resnet18/fold_{args.fold}',
                    resnet50_y_fn = f'./checkpoints/classification/resnet50/mip_y_pet_resnet50/fold_{args.fold}',
                    resnet50_x_fn = f'./checkpoints/classification/resnet50/mip_x_pet_resnet50/fold_{args.fold}',
                    resnet18_x_debrain_fn = f'./checkpoints/classification/debrain/resnet18/mip_x_pet_resnet18/fold_{args.fold}',
                    resnet18_y_debrain_fn = f'./checkpoints/classification/debrain/resnet18/mip_y_pet_resnet18/fold_{args.fold}',
                    resnet50_x_debrain_fn = f'./checkpoints/classification/debrain/resnet50/mip_x_pet_resnet50/fold_{args.fold}',
                    resnet50_y_debrain_fn = f'./checkpoints/classification/debrain/resnet50/mip_y_pet_resnet50/fold_{args.fold}',
                    args=args)
            print(f"Using an ensemble of classifiers")


    if args.load_weights is not None and args.load_weights_second_model is not None and args.pretrained: # initialize the ct and pet branches with pretrained weights
        print('-------\n')
        print(f'Loading weights from {args.load_weights} with PRE-TRAINED CT and PET backbones.')
        print('-------')
        net.load_pretrained_transference(args.load_weights, args.load_weights_second_model)
        return net, net_2

    if args.load_weights is not None:
        print('-------\n')
        print(f'Loading weights from {args.load_weights}')
        print('-------')
        net.load_pretrained_unequal(args.load_weights) # ignore layers with size mismatch - needed when changing output channels
        if args.load_weights_second_model is not None:
            print('-------\n')
            print(f'Loading weights from {args.load_weights_second_model}')
            print('-------')
            net_2.load_pretrained_unequal(args.load_weights_second_model) # ignore layers with size mismatch - needed when changing output channels
    elif args.load_best_val_weights is not None:
        paths = sorted([el for el in os.listdir(args.load_best_val_weights) if 'best' in el and ('.pt' in el or '.pth' in el)])


        if args.load_keyword is not None:
            paths = sorted([el for el in os.listdir(args.load_best_val_weights) if args.load_keyword in el and ('.pt' in el or '.pth' in el)])
        path = os.path.join(args.load_best_val_weights, paths[-1])
        print('-------\n')
        print(f'Loading weights from {path}')
        print('-------')

        net.load_pretrained_unequal(path) # ignore layers with size mismatch - needed when changing output channels

        if args.load_weights_second_model is not None:
            paths = sorted([el for el in os.listdir(args.load_weights_second_model) if 'best' in el])
            if args.load_keyword is not None:
                paths = sorted([el for el in os.listdir(args.load_weights_second_model) if args.load_keyword in el])
            path = os.path.join(args.load_weights_second_model, paths[-1])
            print('-------\n')
            print(f'Loading weights from {path}')
            print('-------')
            net_2.load_pretrained_unequal(path) # ignore layers with size mismatch - needed when changing output channels

    return net, net_2
