import cv2
import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from matplotlib import pyplot as plt
import cc3d

# IGNITE
from ignite.engine import (
    _prepare_batch,
)

# MONAI
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import compute_meandice
from monai.transforms import AddChannel, RandGaussianNoise, RandCoarseShuffle, AsDiscrete, Lambda

# TORCH
import torch
import torch.nn.functional as nnf

from utils.utils import save_nifti_img, brats_post_label_core, brats_post_label_edema, brats_post_pred_core, brats_post_pred_edema

def f_class_label(class_label, batch):
    if class_label != 0:
        class_label = torch.ones(batch['ct_vol'].shape[1:])
    else:
        class_label = torch.zeros(batch['ct_vol'].shape[1:])
    return class_label

def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


# Ignite expects a (input, label) tuple
def prepare_batch(batch, device=None, input_mod=None, non_blocking=False, task=None, args=None):
    if not args.mask_attention:
        inp = batch[input_mod]
    else:
        # Append the attention tensor to all volumes in the batch
        inp = torch.cat((torch.cat(batch[input_mod].shape[0] * [attention]), batch[input_mod]), dim=1) # dim=1 is the channel
    if args.dataset == 'BraTS':
        inp = torch.cat([inp[:,3:], inp[:,:1]], dim=1) # T2w + FLAIR
        label = batch["label"]
        core = label[:, 2:]
        edema =  label[:,:1]
        whole = label[:,1:2]

        core = (core > 0) * 1.0
        edema = (edema > 0) * 1.0
        whole = (whole > 0) * 1.0
        #edema = whole - core

        seg = torch.cat([core, whole, edema], dim=1) # Core, Edema

        return _prepare_batch((inp, seg), device, non_blocking)
    if args.self_supervision != 'L2':
        rand_noise = RandGaussianNoise(prob=1.0, std=0.3)
        rand_shuffle = RandCoarseShuffle(prob=1.0, spatial_size=16, holes=args.n_masks)
    ### Segmentation ###
    if task == 'segmentation':
        return _prepare_batch((inp, batch["seg"]), device, non_blocking)
    ### Reconstruction ###
    elif task == 'reconstruction':
        return _prepare_batch((inp, batch[input_mod]), device, non_blocking)
    ### Classification ###
    elif task == 'classification' and not args.mask_attention and args.sliding_window: # classification without mask, with sliding window inference
        seg = batch[f"mip_seg_{args.proj_dim}"] # png with MIP-GT-mask
        label = []
        for el in seg: # Iterate over batch
            label.append((torch.sum(el) > 0) * 1.0) # pos if at least one voxel has a tumor
        label = torch.Tensor(label)
        return _prepare_batch((inp, label.unsqueeze(dim=1).float()), device, non_blocking)
    elif task == 'classification' and not args.mask_attention and args.proj_dim is None: # classification without mask, with 3 channels (X,Y,Z)
        return _prepare_batch((inp, batch["class_label"].unsqueeze(dim=1).float()), device, non_blocking)
    elif task == 'classification' and args.proj_dim is not None: # classification with 1 channel
        return _prepare_batch((inp, batch["class_label"].unsqueeze(dim=1).float()), device, non_blocking)
    ### Multi-Task Settings ###
    elif task in ['transference', 'fission', 'fission_classification']:
        ct_vol = inp[:, :1]
        cls = None
        if task == 'fission_classification': # Get class
            cls = torch.ones(ct_vol.shape).to(device)
            cls *= batch['class_label'][..., None, None, None, None].to(device)
        if args.self_supervision == 'L2':
            if args.task == 'transference':
                return _prepare_batch((inp, torch.cat([ct_vol, batch['seg']], dim=1)), device, non_blocking)
            elif args.task == 'fission':
                return _prepare_batch((inp, torch.cat([inp, batch['seg']], dim=1)), device, non_blocking)
            elif args.task == 'fission_classification':
                return _prepare_batch((inp, torch.cat([inp, batch['seg'], cls], dim=1)), device, non_blocking)

        elif args.self_supervision == 'L2_noise':
            ct_vol_noisy = rand_noise(ct_vol)
            if args.task == 'transference':
                return _prepare_batch((torch.cat([ct_vol_noisy, inp[:,1:]], dim=1), torch.cat([ct_vol, batch['seg']], dim=1)), device, non_blocking)
            elif args.task == 'fission':
                return _prepare_batch((torch.cat([ct_vol_noisy, inp[:,1:]], dim=1), torch.cat([inp, batch['seg']], dim=1)), device, non_blocking)
            elif args.task == 'fission_classification':
                return _prepare_batch((torch.cat([ct_vol_noisy, inp[:,1:]], dim=1), torch.cat([inp, batch['seg'], cls], dim=1)), device, non_blocking)

        elif args.self_supervision == 'L2_mask':

            ct_vol_masked = ct_vol.clone().detach()

            if args.task == 'transference':
                return _prepare_batch((torch.cat([ct_vol_masked, inp[:,1:]], dim=1), torch.cat([ct_vol, batch['seg']], dim=1)), device, non_blocking)
            elif args.task == 'fission':
                return _prepare_batch((torch.cat([ct_vol_masked, inp[:,1:]], dim=1), torch.cat([inp, batch['seg']], dim=1)), device, non_blocking)
            elif args.task == 'fission_classification':
                return _prepare_batch((torch.cat([ct_vol_masked, inp[:,1:]], dim=1), torch.cat([inp, batch['seg'], cls], dim=1)), device, non_blocking)
        # Ablation study for this paper:
        # Multi-modal Learning from Unpaired Images: Application to Multi-organ Segmentation in CT and MRI, WACV 2018, Vilindria et al.
        elif task == 'alt_transference': # Alternating Transference
            ct_vol = inp[:, :1]
            pet_vol = inp[:, 1:]
            cls = torch.ones(ct_vol.shape).to(device) # Encode modality index - 0: CT, 1: PET

            if np.random.choice([True, False], p=[0.0, 1.0]):
                return _prepare_batch((torch.cat([cls * 0, ct_vol], dim=1), batch["seg"]), device, non_blocking)
            else:
                return _prepare_batch((torch.cat([cls * 1, pet_vol], dim=1), batch["seg"]), device, non_blocking)
        else:
            print('[ERROR] No such self-supervision task is defined.')

    else:
        print("[ERROR]: No such task exists...")
        exit()

def classification(evaluator, val_loader, net, args, device, input_mod=None):
    net.eval()
    outputs, labels = [], []

    evaluator.run(val_loader)
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, device=device, input_mod=input_mod, task='classification', args=args)

            out = net(inp).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            outputs += list(out.flatten())
            labels += list(label.flatten())

    print("Accuracy:", round(np.sum((np.array(outputs) > 0.5) * 1.0 == np.array(labels)) / len(labels), 3))
    net.train()
    if args.evaluate_only:
        np.save(os.path.join(args.log_dir, 'outputs.npy'), np.array(outputs))
        np.save(os.path.join(args.log_dir, 'labels.npy' ), np.array(labels))
    net.train()
    return

def segmentation(evaluator, val_loader, net, args, post_pred, post_label, device, input_mod=None, trainer=None, writer=None):
    if not args.sliding_window and not args.save_nifti:
        evaluator.run(val_loader)
    dices_bg, dices_fg = [], []
    net.eval()

    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, device=device, input_mod=input_mod, args=args, task='segmentation')

            label = (label > 0) * 1.0

            if args.sliding_window:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=False)
            else:
                out = net(inp)

            if args.save_nifti:
                save_nifti(inp, device, out, args, post_pred, post_label, label)

            out = torch.stack([post_pred(i) for i in decollate_batch(out)])
            label = torch.stack([post_label(i) for i in decollate_batch(label)])


            if not args.separate_outputs:
                dice = compute_meandice(out, label, include_background=True)
                dice_F1 = compute_meandice(out, label, include_background=False)

                dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
                dices_bg += list(dice.cpu().detach().numpy().flatten())

    f1_dice = np.nanmean(np.array(dices_fg))
    bg_dice = np.nanmean(np.array(dices_bg))
    if args.learnable_th:
        print('Decision Threshold', net.learnable_th.cpu().detach().numpy())
        writer.add_scalar('Learnable Threshold', net.learnable_th.cpu().detach().numpy(), trainer.state.iteration)

    net.train()
    print('Mean dice foreground:', f1_dice)
    print('Mean dice background:', bg_dice)
    return f1_dice, bg_dice


def transference(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None, writer=None, trainer=None):
    if not args.sliding_window and not args.save_nifti:
        evaluator.run(val_loader)

    dices_fg, dices_bg, rec_loss = [], [], []
    pet_vals = []
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, device=device, input_mod=input_mod, args=args, task='transference')

            label_seg = (label[:,1:] > 0) * 1.0
            label = torch.concat([label[:,:1], label_seg], dim=1)

            if args.sliding_window:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                mask_out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=False)
            else:
                mask_out = net(inp)
            if args.save_nifti:
                save_nifti(inp, device, mask_out, args, post_pred, post_label, label)
            recon_out = mask_out[:, :1]
            rec_loss.append(torch.nanmean(torch.abs(inp[:, :1] - recon_out)).cpu().detach().numpy())

            mask_out = torch.stack([post_pred(i) for i in decollate_batch(mask_out)])
            label = torch.stack([post_label(i) for i in decollate_batch(label)])

            dice = compute_meandice(mask_out, label, include_background=True, ignore_empty=False)
            dice_F1 = compute_meandice(mask_out, label, include_background=False, ignore_empty=False)

            dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
            dices_bg += list(dice.cpu().detach().numpy().flatten())


    dice_F1 = np.nanmean(np.array(dices_fg))
    dice_bg = np.nanmean(np.array(dices_bg))
    r_loss = np.nanmean(np.array(rec_loss))
    print('Mean dice foreground:', dice_F1)
    print('Mean dice background:', dice_bg)
    print('Mean reconstruction loss:', r_loss)

    writer.add_scalar('Reconstruction Loss:', r_loss, trainer.state.iteration)

    net.train()
    return dice_F1, dice_bg
def braTS_eval(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None, writer=None, trainer=None):
    #if not args.sliding_window and not args.save_nifti:
    #    evaluator.run(val_loader)
    post_label_core = Lambda(func=lambda x: brats_post_label_core(x))
    post_label_edema = Lambda(func=lambda x: brats_post_label_edema(x))
    post_pred_core = Lambda(func=lambda x: brats_post_pred_core(x))
    post_pred_edema = Lambda(func=lambda x: brats_post_pred_edema(x))



    discrete = AsDiscrete(argmax=True, to_onehot=2)

    dices_fg, dices_bg, = [], []
    dices_edema = []
    dices_core = []
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, device=device, input_mod=input_mod, args=args, task='none')

            mask_out = net(inp)


            mask_core = torch.stack([post_pred_core(i) for i in decollate_batch(mask_out)])
            label_core = torch.stack([post_label_core(i) for i in decollate_batch(label)])

            mask_edema = torch.stack([post_pred_edema(i) for i in decollate_batch(mask_out)])
            label_edema = torch.stack([post_label_edema(i) for i in decollate_batch(label)])

            mask_out = torch.stack([post_pred(i) for i in decollate_batch(mask_out)])
            mask_out = (mask_out > 0) * 1.0
            label_whole = torch.stack([post_label(i) for i in decollate_batch(label)])
            label_core = torch.stack([post_label_core(i) for i in decollate_batch(label)])
            label_edema = torch.stack([post_label_edema(i) for i in decollate_batch(label)])



            if args.save_nifti:
                save_nifti_img(f'{args.log_dir}/brats_label', label_whole[0][1].cpu().detach().numpy())
                save_nifti_img(f'{args.log_dir}/brats_label_edema', label_edema[0][1].cpu().detach().numpy())
                save_nifti_img(f'{args.log_dir}/brats_label_core', label_core[0][1].cpu().detach().numpy())

                save_nifti_img(f'{args.log_dir}/brats_core', mask_core[0][1].cpu().detach().numpy())
                save_nifti_img(f'{args.log_dir}/brats_edema', mask_edema[0][1].cpu().detach().numpy())
                save_nifti_img(f'{args.log_dir}/brats_whole', mask_out[0][1].cpu().detach().numpy())
                save_nifti_img(f'{args.log_dir}/input', inp[0][0].cpu().detach().numpy())
                save_nifti_img(f'{args.log_dir}/input_2', inp[0][1].cpu().detach().numpy())
                exit()


            dice = compute_meandice(mask_out, label, include_background=True)
            dice_F1 = compute_meandice(mask_out, label, include_background=False)

            dice_core = compute_meandice(mask_core, label_core, include_background=False)
            dice_edema = compute_meandice(mask_edema, label_edema, include_background=False)

            dices_core += list(dice_core.cpu().detach().numpy().flatten())
            dices_edema += list(dice_edema.cpu().detach().numpy().flatten())

            dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
            dices_bg += list(dice.cpu().detach().numpy().flatten())

    dice_F1 = np.nanmean(np.array(dices_fg))
    dice_bg = np.nanmean(np.array(dices_bg))
    dice_core = np.nanmean(np.array(dices_core))
    dice_edema = np.nanmean(np.array(dices_edema))
    print('Mean dice foreground:', dice_F1)
    print('Mean dice (tumor core)', dice_core)
    print('Mean dice (tumor edema)', dice_edema)

    writer.add_scalar('Dice Whole Tumor', dice_F1, trainer.state.iteration)
    writer.add_scalar('Dice Core', dice_core, trainer.state.iteration)
    writer.add_scalar('Dice Edema', dice_edema, trainer.state.iteration)


    net.train()
    return dice_F1, dice_bg


def fission(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None, writer=None, trainer=None):
    #if not args.sliding_window and not args.save_nifti:
    #    evaluator.run(val_loader)


    dices_fg, dices_bg, rec_loss = [], [], []
    class_pred, class_label = [], []
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            task = 'fission' if args.task == 'fission' else 'fission_classification'
            (inp, label) = prepare_batch(val_data, device=device, input_mod=input_mod, args=args, task=task)

            if args.sliding_window:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                mask_out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=False)
            else:
                mask_out = net(inp)

            if args.save_nifti:
                save_nifti(inp, device, mask_out, args, post_pred, post_label, label)

            mask_out = torch.stack([post_pred(i) for i in decollate_batch(mask_out)])
            label = torch.stack([post_label(i) for i in decollate_batch(label)])


            recon_out = mask_out[:, :2]
            rec_loss.append(torch.nanmean(torch.abs(inp[:, :2] - recon_out)).cpu().detach().numpy())
            if args.task == 'fission_classification':
                cls_out = mask_out[:, 4:]
                cls_out = torch.stack([torch.nanmean(el) for el in cls_out]).cpu().detach().numpy() # Does not evaluate sliding_window properly

                cls_gt = label[:, 3:]
                cls_gt = torch.stack([torch.nanmean(el) for el in cls_gt]).cpu().detach().numpy()
                for j, el in enumerate(cls_out):
                    class_pred.append((el > 0.5) * 1.0)
                    class_label.append(cls_gt[j])

            dice = compute_meandice(mask_out, label, include_background=True)
            dice_F1 = compute_meandice(mask_out, label, include_background=False)

            dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
            dices_bg += list(dice.cpu().detach().numpy().flatten())

    dice_F1 = np.nanmean(np.array(dices_fg))
    dice_bg = np.nanmean(np.array(dices_bg))
    r_loss = np.nanmean(np.array(rec_loss))
    acc = np.sum(np.array(class_pred) == np.array(class_label)) / len(class_pred)
    print('Mean dice foreground:', dice_F1)
    print('Mean dice background:', dice_bg)
    print('Mean reconstruction loss:', r_loss)
    print('Mean classification accuracy', acc)
    writer.add_scalar('Dice F1:', dice_F1, trainer.state.iteration)
    writer.add_scalar('Reconstruction Loss:', r_loss, trainer.state.iteration)
    writer.add_scalar('Reconstruction Loss:', r_loss, trainer.state.iteration)
    writer.add_scalar('Classification Accuracy:', acc, trainer.state.iteration)

    net.train()
    return dice_F1, dice_bg

def reconstruction(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None, writer=None, trainer=None):
    if not args.sliding_window and not args.save_nifti:
        evaluator.run(val_loader)

    rec_loss = []
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, device=device, input_mod=input_mod, args=args, task='reconstruction')

            if args.sliding_window:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                mask_out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=False)
            else:
                mask_out = net(inp)
            recon_out = mask_out
            rec_loss.append(torch.nanmean(torch.abs(inp - recon_out)).cpu().detach().numpy())

            if args.save_nifti:
                save_nifti(inp, device, out, args, post_pred, post_label, label)

    r_loss = np.nanmean(np.array(rec_loss))

    print('Mean reconstruction loss:', r_loss)
    writer.add_scalar('Reconstruction Loss:', r_loss, trainer.state.iteration)

    net.train()
    return r_loss


def segmentation_late_fusion(evaluator, val_loader, net, net_2, args, post_pred, post_label, device, input_mod=None):
    # Assume input_mod_1 = ct_vol, input_mod_2 = pet_vol
    dices_fg, dices_bg = [], []
    net.eval()
    net_2.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, device=device, input_mod='ct_pet_vol', task='segmentation', args=args)

            inp_1 = inp[:, :1]
            inp_2 = inp[:, 1:]

            if args.sliding_window:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                out_1 = sliding_window_inference(inp_1, roi_size, sw_batch_size, net, progress=False)
                out_2 = sliding_window_inference(inp_2, roi_size, sw_batch_size, net_2, progress=False)
            else:
                out_1 = net(inp_1)
                out_2 = net_2(inp_2)

            if args.logit_fusion:
                out = (out_1 + out_2) / 2
                out = torch.stack([post_pred(i) for i in decollate_batch(out)])
            elif args.decision_fusion:
                out_1 = torch.stack([post_pred(i) for i in decollate_batch(out_1)])
                out_2 = torch.stack([post_pred(i) for i in decollate_batch(out_2)])
                out = (out_1 + out_2) / 2
                out = (out >= 0.5) * 1.0
            else:
                print('[ERROR] Only logit and decision fusion are implemented!')
                exit()

            label = torch.stack([post_label(i) for i in decollate_batch(label)])

            dice = compute_meandice(out, label, include_background=True)
            dice_F1 = compute_meandice(out, label, include_background=False)

            dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
            dices_bg += list(dice.cpu().detach().numpy().flatten())

    f1_dice = np.nanmean(np.array(dices_fg))
    bg_dice = np.nanmean(np.array(dices_bg))
    print('Mean dice foreground:', f1_dice)
    print('Mean dice background:', bg_dice)
    suffix = 'decision' if args.decision_fusion else 'logit'
    with open(os.path.join(args.log_dir, f'late_fusion_{suffix}.txt'), 'w') as f:
        f.write(f'f1 dice: {f1_dice}, bg_dice: {bg_dice}')
    return f1_dice, bg_dice


# Utility for save_nifti()
def convert_output(out, device, spatial_size, add_channel):
    out = torch.Tensor(out.cpu().detach().numpy()).to(device)
    out = nnf.interpolate(add_channel(add_channel(out[0, 1])), size=spatial_size)[0][0]
    return out

# Utility for save_nifti()
def write_nifti(names, data, affine, args):
    for i, im in enumerate(data):
        ni_img = nib.Nifti1Image(im.cpu().detach().numpy(), affine=affine)
        ni_img.header.get_xyzt_units()
        ni_img.to_filename(f'{args.log_dir}/{names[i]}.nii.gz')

# Save Nifti for visualization
def save_nifti(inp, device, out, args, post_pred, post_label, label):
    add_channel = AddChannel()
    affine = np.eye(4)
    affine[0][0] = -1
    spatial_size = (400, 400, 384) # Fix the spatial size to simplify visualization
    if args.separate_outputs:
        if args.task not in ['fission', 'fission_classification']:
            (out_ct, out_pet) = out
        else:
            (out_ct, out_pet, out_seg) = out
        if args.task == 'segmentation':
            out_fused = torch.stack([post_pred(i) for i in decollate_batch(out_ct * args.mirror_th + out_pet * (1.0 - args.mirror_th))])
            out_fused = convert_output(out_fused, device, spatial_size, add_channel)

        if args.task not in ['transference', 'fission', 'fission_classification']:
            out_ct = torch.stack([post_pred(i) for i in decollate_batch(out_ct)])

        if args.task not in ['fission', 'fission_classification']:
            out_pet = torch.stack([post_pred(i) for i in decollate_batch(out_pet)])
        else:
            out_seg = torch.stack([post_pred(i) for i in decollate_batch(out_seg)])
        label = torch.stack([post_label(i) for i in decollate_batch(label)])

        inp_ct = torch.Tensor(inp.cpu().detach().numpy()).to(device)
        inp_ct = nnf.interpolate(add_channel(add_channel(inp_ct[0][0])), size=spatial_size)[0][0]

        inp_pet = convert_output(inp, device, spatial_size, add_channel)

        if args.task not in ['transference', 'fission', 'fission_classification']:
            out_ct = convert_output(out_ct, device, spatial_size, add_channel)
        else:
            out_ct = nnf.interpolate(add_channel(add_channel(out_ct[0, 0])), size=spatial_size)[0][0]
            out_ct *= ((inp_ct > 0) * 1.0)

        if args.task not in ['fission', 'fission_classification']:
            out_pet = convert_output(out_pet, device, spatial_size, add_channel)
        else:
            out_pet = nnf.interpolate(add_channel(add_channel(out_pet[0, 0])), size=spatial_size)[0][0]
            out_pet *= ((inp_pet > 0) *1.0)
            out_seg = convert_output(out_seg, device, spatial_size, add_channel)


        out_label = convert_output(label, device, spatial_size, add_channel)


        names = ['ct', 'pet', 'ct_pred', 'pet_pred', 'label']
        data = [inp_ct, inp_pet, out_ct, out_pet, out_label]

        if args.task in ['fission', 'fission_classification']:
            names.append('out_seg')
            data.append(out_seg)

        if args.task == 'segmentation':
            names.append('fused_pred')
            data.append(out_fused)

        write_nifti(names, data, affine, args)

    elif args.single_mod is not None or args.task == 'reconstruction':
        inp_ct = torch.Tensor(inp.cpu().detach().numpy()).to(device)
        inp_ct = nnf.interpolate(add_channel(add_channel(inp_ct[0][0])), size=spatial_size)[0][0]

        out = convert_output(out, device, spatial_size, add_channel)
        out_label = convert_output(label, device, spatial_size, add_channel)

        names = ['inp', 'ct_pred', 'label']
        data = [inp_ct, out, out_label]
        write_nifti(names, data, affine, args)

    else: # Early fusion or Transference
        inp_ct = torch.Tensor(inp.cpu().detach().numpy()).to(device)
        inp_ct = nnf.interpolate(add_channel(add_channel(inp_ct[0][0])), size=spatial_size)[0][0]

        inp_pet = convert_output(inp, device, spatial_size, add_channel)


        out = torch.stack([post_pred(i) for i in decollate_batch(out)])
        label = torch.stack([post_label(i) for i in decollate_batch(label)])

        out_pred = convert_output(out, device, spatial_size, add_channel) # Mask Prediction
        out_label = convert_output(label, device, spatial_size, add_channel)

        names = ['inp_ct', 'inp_pet', 'pred', 'label']
        data = [inp_ct, inp_pet, out_pred, out_label]
        write_nifti(names, data, affine, args)

    exit()
