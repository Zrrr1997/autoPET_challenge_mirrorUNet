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
from monai.metrics import compute_meandice
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

def classification_sliding_window(args, val_loader, net, evaluator, device, input_mod=None):
	if args.evaluate_only:
		correct, incorrect = 0, 0
		all_preds, all_logits, all_labels = [], [], []

		print('Calculating sliding window inference. This might take a while...')
		with torch.no_grad():
			for i, val_data in tqdm(enumerate(val_loader)):
				roi_size = (100, 100)
				sw_batch_size = 1

				(inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)

				preds = []
				all_labels += list(label.cpu().detach().numpy().flatten())

				for img in inp:
					img_pred = 0
					logits = []
					for w in range(4):
						for h in range(4):
							img_ = img[0][100 * h : 100 * (h + 1), 100 * w : 100 * (w + 1)].unsqueeze(0).unsqueeze(0) # remove channel and add BxC dims again

							logit = net(img_)
							logits.append(logit.cpu().detach().numpy())
							pred = (logit > 0.5) * 1.0

							img_pred += pred
							preds.append((img_pred > 0) * 1.0)
							all_logits.append(logits)

						all_preds += list([el.cpu().detach().numpy() for el in preds])

						preds = torch.Tensor(preds).to(device)
						correct += torch.sum((preds.flatten() == label.flatten()) * 1.0)
						incorrect += torch.sum((preds != label) * 1.0)
		np.save(os.path.join(args.log_dir, 'preds.npy'), np.array(all_preds))
		np.save(os.path.join(args.log_dir, 'scores.npy'), np.array(all_logits))
		np.save(os.path.join(args.log_dir, 'labels.npy'), np.array(all_labels))
		print('Whole Image Accuracy:', correct / (correct + incorrect))
		return # done with evaluation...
	else:
		evaluator.run(val_loader)
	return


def segmentation_sliding_window(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None):
        dices_fg, dices_bg = [], []
        with torch.no_grad():
            for i, val_data in tqdm(enumerate(val_loader)):
                roi_size = (96, 96, 96)
                sw_batch_size = 4

                (inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)

                mask_out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=False)

                mask_out = torch.stack([post_pred(i) for i in decollate_batch(mask_out)])

                label = torch.stack([post_label(i) for i in decollate_batch(label)])


                if i == 0 and args.save_eval_img: # 64th slice might not contain a tumor !
                    mask_img_bg = mask_out[0, 0, :, :, 64] * 255
                    mask_img_fg = mask_out[0, 1, :, :, 64] * 255
                    ct = inp[0, 0, :, :, 64] * 255
                    pet = inp[0, 1, :, :, 64] * 255

                    gt = label[0, 1, :, :, 64] * 255
                    cv2.imwrite(os.path.join(args.log_dir, 'mask_img_bg.png'), mask_img_bg.cpu().detach().numpy())
                    cv2.imwrite(os.path.join(args.log_dir, 'mask_img_fg.png'), mask_img_fg.cpu().detach().numpy())
                    cv2.imwrite(os.path.join(args.log_dir, 'ct.png'), ct.cpu().detach().numpy())
                    cv2.imwrite(os.path.join(args.log_dir, 'pet.png'), pet.cpu().detach().numpy())

                    cv2.imwrite(os.path.join(args.log_dir, 'gt.png'), gt.cpu().detach().numpy()) # might be empty

                dice = compute_meandice(label, mask_out, include_background=True)
                dice_F1 = compute_meandice(label, mask_out, include_background=False)


                dices_fg.append(dice_F1.cpu().detach().numpy())
                dices_bg.append(dice.cpu().detach().numpy())

        f1_dice = np.mean(np.array(dices_fg))
        bg_dice = np.mean(np.array(dices_bg))
        print('Mean dice foreground:', f1_dice)
        print('Mean dice background:', bg_dice)

        return  f1_dice, bg_dice

def transference_sliding_window(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None):
        dices_fg, dices_bg = [], []
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                roi_size = (96, 96, 96)
                sw_batch_size = 4

                (inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)

                mask_out = sliding_window_inference(inp, roi_size, sw_batch_size, net, progress=True)
                recon_out = mask_out[:, :1]
                print('Reconstruction loss in transference:', torch.mean(torch.abs(inp[:, :1] - recon_out)))

                if args.save_eval_img:
                    ct = inp[0, 0, :, :, 64] * 255
                    ct_recon = recon_out[0, 0, :, :, 64] * 255
                    cv2.imwrite(os.path.join(args.log_dir, 'ct.png'), ct.cpu().detach().numpy())
                    cv2.imwrite(os.path.join(args.log_dir, 'ct_recon.png'), ct_recon.cpu().detach().numpy())


                mask_out = mask_out[:, 1:]
                mask_out = torch.stack([post_pred(i) for i in decollate_batch(mask_out)])
                label = torch.stack([post_label(i) for i in decollate_batch(label)])

                dice = compute_meandice(label, mask_out, include_background=True)
                dice_F1 = compute_meandice(label, mask_out, include_background=False)


                dices_fg.append(dice_F1.cpu().detach().numpy())
                dices_bg.append(dice.cpu().detach().numpy())

        print('Mean dice foreground:', np.mean(np.array(dices_fg)))
        print('Mean dice background:', np.mean(np.array(dices_bg)))

        return


def classification_normal(evaluator, val_loader, net, args, device, input_mod=None):
    net.eval()
    outputs = []
    labels = []

    evaluator.run(val_loader)
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
    return

# Normal Segmentation without sliding window
def segmentation_normal(evaluator, val_loader, net, args, post_pred, post_label, device, input_mod=None):
    evaluator.run(val_loader)
    dices_bg, dices_fg = [], []
    net.eval()
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)
            out = net(inp)

            out = torch.stack([post_pred(i) for i in decollate_batch(out)])
            label = torch.stack([post_label(i) for i in decollate_batch(label)])

            dice = compute_meandice(label, out, include_background=True)
            dice_F1 = compute_meandice(label, out, include_background=False)

            dices_fg += list(dice_F1.cpu().detach().numpy().flatten())
            dices_bg += list(dice.cpu().detach().numpy().flatten())

        f1_dice = np.mean(np.array(dices_fg))
        bg_dice = np.mean(np.array(dices_bg))
    net.train()
    print('Mean dice foreground:', f1_dice)
    print('Mean dice background:', bg_dice)
    return


def transference_normal(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=None, writer=None, trainer=None):

    evaluator.run(val_loader)

    dices_fg, dices_bg, rec_loss = [], [], []
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            (inp, label) = prepare_batch(val_data, args, device=device, input_mod=input_mod)
            mask_out = net(inp)
            recon_out = mask_out[:, :1]
            rec_loss.append(torch.mean(torch.abs(inp[:, :1] - recon_out)).cpu().detach().numpy())


            mask_out = torch.stack([post_pred(i) for i in decollate_batch(mask_out)])
            label = torch.stack([post_label(i) for i in decollate_batch(label)])


            dice = compute_meandice(label, mask_out, include_background=True)
            dice_F1 = compute_meandice(label, mask_out, include_background=False)

            dices_fg.append(dice_F1.cpu().detach().numpy().flatten())
            dices_bg.append(dice.cpu().detach().numpy().flatten())

    print('Mean dice foreground:', np.mean(np.array(dices_fg)))
    print('Mean dice background:', np.mean(np.array(dices_bg)))
    print('Mean reconstruction loss:', np.mean(np.array(rec_loss)))
    writer.add_scalar('Reconstruction Loss:', np.mean(np.array(rec_loss)), trainer.state.iteration)


    return

# not implemented
def segmentation_classification():
    # TODO not implemented
    outputs = []
    outputs_class = []
    labels = []
    labels_class = []
    args_2 = args
    args_2.task = 'classification'
    args_2.load_weights = './checkpoints/PET/resnet-50-pet/net_best_val.pt'
    args_2.resnet_version = 'resnet50'

    class_net = prepare_model(device=device, out_channels=1, args=args_2)
    with torch.no_grad():
        for i, val_data in tqdm(enumerate(val_loader)):
            (inp, label) = prepare_batch(val_data, args, device=device, task='segmentation')
            out_seg = net(inp)
            label_seg = label.cpu().detach().numpy()
            (inp, label) = prepare_batch(val_data, args, device=device, task='classification')
            label_class = label.cpu().detach().numpy()
            out_class = class_net(inp)
            out_class *= (out_class > 0.5) # Binarize

            out_class = torch.argmax(out_class, dim=1, keepdim=True)

            out_seg[:,1,:,:,:] *= out_class.unsqueeze(dim=-1).unsqueeze(dim=-1) # multiply Fg mask with class prediction

            outputs += list(out_seg.cpu().detach().numpy().flatten())
            labels += list(label_seg.flatten())
            outputs_class += list(out_class.cpu().detach().numpy().flatten())
            labels_class += list(label_class.flatten())
    np.save(os.path.join(args.log_dir, 'outputs.npy'), np.array(outputs).reshape(-1, 2, 400, 400, 128))
    np.save(os.path.join(args.log_dir, 'labels.npy' ), np.array(labels).reshape(-1, 1, 400, 400, 128))
    np.save(os.path.join(args.log_dir, 'outputs_class.npy'), np.array(outputs_class))
    np.save(os.path.join(args.log_dir, 'labels_class.npy' ), np.array(labels_class))
    return
