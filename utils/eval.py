import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import nibabel as nib

from ignite.engine import (
    _prepare_batch,
)
# MONAI
from monai.inferers import sliding_window_inference
from monai.data import list_data_collate, decollate_batch
from monai.metrics import compute_meandice, compute_meaniou
from monai.visualize import GradCAM


# Auxiliary method to load (sample, label) depending on the task and configuration
def prepare_batch(batch, args, device=None, non_blocking=False, input_mod=None):
    task = args.task
    # Handle attention mask
    if not args.mask_attention:
        inp = batch[input_mod]
    else:
        inp = torch.cat((torch.cat(batch[input_mod].shape[0] * [attention]), batch[input_mod]), dim=1) # dim=1 is the channel

    if task == 'segmentation' or task == 'segmentation_classification' or args.class_backbone == 'Ensemble':
        return _prepare_batch((inp, batch["seg"]), device, non_blocking)
    elif task == 'reconstruction':
        return _prepare_batch((inp, batch[input_mod]), device, non_blocking)
    elif task == 'classification' and not args.mask_attention and args.sliding_window: # classification without mask, with sliding window inference
        seg = batch[f"mip_seg_{args.proj_dim}"]
        label = []
        for el in seg:
            label.append((torch.sum(el) > 0) * 1.0) # Label must be computed manually for each patch...
        label = torch.Tensor(label)
        return _prepare_batch((inp, label.unsqueeze(dim=1).float()), device, non_blocking)
    elif task == 'classification' and args.proj_dim is not None: # normal classification
        return _prepare_batch((inp, batch["class_label"].unsqueeze(dim=1).float()), device, non_blocking)
    elif task == 'transference':
        ct_vol = inp[:, :1]
        return _prepare_batch((inp, torch.cat([ct_vol, batch['seg']], dim=1)), device, non_blocking)
        #return _prepare_batch((inp, torch.cat([batch['ct_vol'], batch['seg']], dim=1)), device, non_blocking)
    else:
        print("[ERROR]: No such task exists...")
        exit()

def classification(evaluator, val_loader, net, args, device, input_mod=None):
    net.eval()
    outputs = []
    labels = []

    evaluator.run(val_loader)
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)

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
    if not args.sliding_window:
        evaluator.run(val_loader)
    dices_bg, dices_fg = [], []
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)

            if args.sliding_window:
                roi_size = (96, 96, 96)
                sw_batch_size = 4

                out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=False)
            else:
                out = net(inp)

            out = torch.stack([post_pred(i) for i in decollate_batch(out)])
            label = torch.stack([post_label(i) for i in decollate_batch(label)])

            dice = compute_meandice(out, label, include_background=True)
            dice_F1 = compute_meandice(out, label, include_background=False)

            dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
            dices_bg += list(dice.cpu().detach().numpy().flatten())

    f1_dice = np.mean(np.array(dices_fg))
    bg_dice = np.mean(np.array(dices_bg))
    if args.learnable_th:
        print('Decision Threshold', net.learnable_th.cpu().detach().numpy())
        writer.add_scalar('Learnable Threshold', net.learnable_th.cpu().detach().numpy(), trainer.state.iteration)


    net.train()
    print('Mean dice foreground:', f1_dice)
    print('Mean dice background:', bg_dice)
    return f1_dice, bg_dice


def transference(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None, writer=None, trainer=None):

    if not args.sliding_window:
        evaluator.run(val_loader)

    dices_fg, dices_bg, rec_loss = [], [], []
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)

            if args.sliding_window:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                mask_out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=False)
            else:
                mask_out = net(inp)
            recon_out = mask_out[:, :1]
            rec_loss.append(torch.mean(torch.abs(inp[:, :1] - recon_out)).cpu().detach().numpy())


            mask_out = torch.stack([post_pred(i) for i in decollate_batch(mask_out)])
            label = torch.stack([post_label(i) for i in decollate_batch(label)])

            dice = compute_meandice(mask_out, label, include_background=True)
            dice_F1 = compute_meandice(mask_out, label, include_background=False)

            dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
            dices_bg += list(dice.cpu().detach().numpy().flatten())


    dice_F1 = np.mean(np.array(dices_fg))
    dice_bg = np.mean(np.array(dices_bg))
    r_loss = np.mean(np.array(rec_loss))
    print('Mean dice foreground:', dice_F1)
    print('Mean dice background:', dice_bg)
    print('Mean reconstruction loss:', r_loss)
    writer.add_scalar('Reconstruction Loss:', r_loss, trainer.state.iteration)

    net.train()
    return dice_F1, dice_bg
def segmentation_late_fusion(evaluator, val_loader, net, net_2, args, post_pred, post_label, device, input_mod=None):
    # Assume input_mod_1 = ct_vol, input_mod_2 = ct_pet_vol
    dices_fg, dices_bg = [], []
    net.eval()
    net_2.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, args, device=device, input_mod='ct_pet_vol')


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

    f1_dice = np.mean(np.array(dices_fg))
    bg_dice = np.mean(np.array(dices_bg))
    print('Mean dice foreground:', f1_dice)
    print('Mean dice background:', bg_dice)
    suffix = 'decision' if args.decision_fusion else 'logit'
    with open(os.path.join(args.log_dir, f'late_fusion_{suffix}.txt'), 'w') as f:
        f.write(f'f1 dice: {f1_dice}, bg_dice: {bg_dice}')
    return f1_dice, bg_dice
# not implemented
def segmentation_classification():
    pass
