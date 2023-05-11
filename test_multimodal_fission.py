import os
import sys
import json
import logging
import argparse

# Torch
import torch
from torch.utils.tensorboard import SummaryWriter

# IGNITE
import ignite
from ignite.engine import (
    Events,
    _prepare_batch,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import ModelCheckpoint

# MONAI
from monai.data import decollate_batch

from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    LrScheduleHandler,
)
from monai.transforms import RandGaussianNoise, RandCoarseShuffle

# Utils
from utils.data_utils import prepare_loaders
from utils.parser import prepare_parser
from utils.models import prepare_model
from utils.utils import *

# Eval
from utils.eval import *


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Use all available cores
    print(f"CPU Count: {os.cpu_count()}")
    torch.set_num_threads(os.cpu_count())
    print(f"Num threads: {torch.get_num_threads()}")

    parser = argparse.ArgumentParser(description='Mirror U-Net for AutoPET: codebase implementation.')
    parser = prepare_parser(parser)
    args = parser.parse_args()
    print('--------\n')
    print(args, '\n')
    print('--------')

    check_args(args)

    out_channels = prepare_out_channels(args) # Output channels per branch
    input_mod = prepare_input_mod(args)

    best_f1_dice, best_bg_dice = 0, 0

    # Create Model, Loss, and Optimizer
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")
    net, net_2 = prepare_model(device=device, out_channels=out_channels, args=args)

    # Data configurations
    spatial_size = [224, 224, 128] if (args.class_backbone == 'CoAtNet' and args.task == 'classification') else [400, 400, 128] # Fix axial resolution for non-sliding window inference
    train_loader, val_loader = prepare_loaders(in_dir=args.in_dir, spatial_size=spatial_size, args=args)

    # Tensorboard stats
    writer = SummaryWriter(log_dir = args.log_dir)

    # Write the current configuration to file
    with open(os.path.join(args.log_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    check_data = first(val_loader)
    print('Input shape check:', check_data[input_mod].shape)

    if args.save_network_graph_image:
        data = check_data['ct_pet_vol'].to(device)
        save_network_graph_plot(net, data, args)

    check_data_shape(val_loader, input_mod, args)

    # Hyperparameters
    loss = prepare_loss(args)
    lr = args.lr

    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=0)

    # Noise or Voxel Shuffling
    if args.self_supervision != 'L2':
        rand_noise = RandGaussianNoise(prob=1.0, std=0.3)
        rand_shuffle = RandCoarseShuffle(prob=1.0, spatial_size=16, holes=args.n_masks)

    # Append prior distribution of training labels to the input
    attention = None
    if args.mask_attention:
        attention = prepare_attention(args)
        print('Attention shape:', attention.shape)

    # Utility function to propagate the class label during inference
    def f_class_label(class_label, batch):
        if class_label != 0:
            class_label = torch.ones(batch.shape[1:])
        else:
            class_label = torch.zeros(batch.shape[1:])
        return class_label

    # Ignite expects a (input, label) tuple
    def prepare_batch(batch, device=None, non_blocking=False, task=args.task):
        if not args.mask_attention:
            inp = batch[input_mod]
            if args.ct_ablation:
                ct_vol = inp[:, :1]
                inp = torch.cat([ct_vol, ct_vol], dim=1)
            if args.pet_ablation:
                assert not args.ct_ablation
                pet_vol = inp[:, 1:]
                inp = torch.cat([pet_vol, pet_vol], dim=1)
        else:
            # Append the attention tensor to all volumes in the batch
            inp = torch.cat((torch.cat(batch[input_mod].shape[0] * [attention]), batch[input_mod]), dim=1) # dim=1 is the channel

        # BraTS ablation
        if args.dataset == 'BraTS':
            inp = torch.cat([inp[:,3:], inp[:,:1]], dim=1) # T2w + FLAIR
            print('Input', inp.shape)
            label = batch["label"]
            core = label[:, 2:]
            edema =  label[:,:1]
            whole = label[:,1:2]

            core = (core > 0) * 1.0
            edema = (edema > 0) * 1.0
            whole = (whole > 0) * 1.0
            #edema = whole - core
            '''
            save_nifti_img('core', core[0,0])
            save_nifti_img('edema', edema[0,0])
            save_nifti_img('whole', whole[0,0])
            save_nifti_img('T2w', inp[:,:1][0,0])
            save_nifti_img('FLAIR', inp[:,1:][0,0])
            exit()
            '''
            seg = torch.cat([core, whole, edema], dim=1) # Core, Edema

            return _prepare_batch((inp, seg), device, non_blocking)
        ### Segmentation ###
        elif task in 'segmentation' or args.dataset == 'BraTS':
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
                ct_vol_masked = rand_shuffle(ct_vol_masked)

                if args.task == 'transference':
                    return _prepare_batch((torch.cat([ct_vol_masked, inp[:,1:]], dim=1), torch.cat([ct_vol, batch['seg']], dim=1)), device, non_blocking)
                elif args.task == 'fission':
                    return _prepare_batch((torch.cat([ct_vol_masked, inp[:,1:]], dim=1), torch.cat([inp, batch['seg']], dim=1)), device, non_blocking)
                elif args.task == 'fission_classification':
                    return _prepare_batch((torch.cat([ct_vol_masked, inp[:,1:]], dim=1), torch.cat([inp, batch['seg'], cls], dim=1)), device, non_blocking)

            else:
                print('[ERROR] No such self-supervision task is defined.')
        # Ablation study for this paper:
        # Multi-modal Learning from Unpaired Images: Application to Multi-organ Segmentation in CT and MRI, WACV 2018, Vilindria et al.
        elif task == 'alt_transference': # Alternating Transference
            ct_vol = inp[:, :1]
            pet_vol = inp[:, 1:]
            cls = torch.ones(ct_vol.shape).to(device) # Encode modality index - 0: CT, 1: PET

            if np.random.choice([True, False], p=[0.5, 0.5]):
                return _prepare_batch((torch.cat([cls * 0, ct_vol], dim=1), batch["seg"]), device, non_blocking)
            else:
                return _prepare_batch((torch.cat([cls * 1, pet_vol], dim=1), batch["seg"]), device, non_blocking)


        else:
            print("[ERROR]: No such task exists...")
            exit()


    # Metric to evaluate whether to save the "best" model or not
    def default_score_fn(engine):
        if args.task == 'classification':
            score = engine.state.metrics['Accuracy']
        elif args.task == 'reconstruction':
            score = engine.state.metrics['MSE']
        else: # Segmentation, Transference, Fission, Alt_Transference
            score = engine.state.metrics['Mean_Dice']
        return score
    def default_score_fn_F1(engine):
        return engine.state.metrics['Mean_Dice_F1'] # Only foreground dice (tumors)

    trainer = create_supervised_trainer(
        net, opt, loss, device, False, prepare_batch=prepare_batch
    )
    checkpoint_handler = ModelCheckpoint(
        args.ckpt_dir, "net", n_saved=20, require_empty=False
    )

    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED(every=args.save_every),
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )



    # Logging
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch
    train_tensorboard_stats_handler = TensorBoardStatsHandler(output_transform=lambda x: x, log_dir=args.log_dir,)
    train_tensorboard_stats_handler.attach(trainer)

    # Learning rate drop-off at every args.lr_step_size epochs
    train_lr_handler = LrScheduleHandler(lr_scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=0.1), print_lr=True)
    train_lr_handler.attach(trainer)

    # Validation configuration
    validation_every_n_iters = args.eval_every

    val_metrics = prepare_val_metrics(args)

    post_pred, post_label = prepare_post_fns(args)

    evaluator = create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        output_transform=lambda x, y, y_pred: ([post_pred(i) for i in decollate_batch(y_pred)], [post_label(i) for i in decollate_batch(y)]),
        prepare_batch=prepare_batch,
    )
    if args.task in ['classification', 'segmentation', 'transference', 'reconstruction', 'fission', 'fission_classification', 'alt_transference']:
        checkpoint_handler_best_val = ModelCheckpoint(
            args.ckpt_dir, "net_best_val", n_saved=1, require_empty=False, score_function=default_score_fn
        )
        if args.task in ['segmentation', 'transference', 'fission', 'fission_classification', 'alt_transference']:
            checkpoint_handler_best_val_F1 = ModelCheckpoint(
                args.ckpt_dir, "net_best_val_F1", n_saved=1, require_empty=False, score_function=default_score_fn_F1
            )
            evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler_best_val_F1, {'net': net, })

        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler_best_val, {'net': net, })


    @trainer.on(Events.ITERATION_COMPLETED(every=validation_every_n_iters))
    def run_validation(engine):

        global best_f1_dice, best_bg_dice

        if args.dataset == 'BraTS':
            f1_dice, bg_dice = braTS_eval(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=input_mod, writer=writer, trainer=trainer)
            writer.add_scalar('F1 Dice', f1_dice, trainer.state.iteration)
            writer.add_scalar('BG Dice', bg_dice, trainer.state.iteration)

            if f1_dice > best_f1_dice:
                torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'best_f1_dice.pth'))
                best_f1_dice = f1_dice
                with open(os.path.join(args.ckpt_dir, f'best_f1_dice.txt'), 'a+') as f:
                    f.write(str(best_f1_dice) + '\n')

            if args.evaluate_only:
                exit()
            return
        #####################################
        ##         CLASSIFICATION          ##
        #####################################
        if args.task=='classification':
            classification(evaluator, val_loader, net, args, device, input_mod=input_mod)
            if args.evaluate_only:
                exit()
        #####################################
        ##         SEGMENTATION            ##
        #####################################
        if args.task in ['segmentation', 'alt_transference']:
            # Late fusion of two networks
            if net_2 is not None and args.load_weights_second_model is not None:
                f1_dice, bg_dice = segmentation_late_fusion(evaluator, val_loader, net, net_2, args, post_pred, post_label, device)
            else:
                f1_dice, bg_dice = segmentation(evaluator, val_loader, net, args, post_pred, post_label, device, input_mod=input_mod, trainer=trainer, writer=writer)
            if args.evaluate_only:
                exit()
            if args.sliding_window:
                writer.add_scalar('F1 Dice', f1_dice, trainer.state.iteration)
                writer.add_scalar('BG Dice', bg_dice, trainer.state.iteration)

                if f1_dice > best_f1_dice:
                    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'best_f1_dice.pth'))
                    best_f1_dice = f1_dice
                    with open(os.path.join(args.ckpt_dir, f'best_f1_dice.txt'), 'a+') as f:
                        f.write(str(best_f1_dice) + '\n')
                if bg_dice > best_bg_dice:
                    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'best_bg_dice.pth'))
                    best_bg_dice = bg_dice
                    with open(os.path.join(args.ckpt_dir, f'best_bg_dice.txt'), 'a+') as f:
                        f.write(str(best_bg_dice) + '\n')
        #####################################
        ##         TRANSFERENCE            ##
        #####################################
        if args.task=='transference':
            f1_dice, bg_dice = transference(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=input_mod, writer=writer, trainer=trainer)
            if args.evaluate_only:
                exit()
            if args.sliding_window:
                writer.add_scalar('F1 Dice', f1_dice, trainer.state.iteration)
                writer.add_scalar('BG Dice', bg_dice, trainer.state.iteration)

                if f1_dice > best_f1_dice:
                    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'best_f1_dice.pth'))
                    best_f1_dice = f1_dice
                    with open(os.path.join(args.ckpt_dir, f'best_f1_dice.txt'), 'a+') as f:
                        f.write(str(best_f1_dice) + '\n')
                if bg_dice > best_bg_dice:
                    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'best_bg_dice.pth'))
                    best_bg_dice = bg_dice
                    with open(os.path.join(args.ckpt_dir, f'best_bg_dice.txt'), 'a+') as f:
                        f.write(str(best_bg_dice) + '\n')
        #####################################
        ##          FISSION                ##
        #####################################
        if args.task in ['fission', 'fission_classification']:
            f1_dice, bg_dice = fission(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=input_mod, writer=writer, trainer=trainer)
            if args.evaluate_only:
                exit()
            if args.sliding_window:
                writer.add_scalar('F1 Dice', f1_dice, trainer.state.iteration)
                writer.add_scalar('BG Dice', bg_dice, trainer.state.iteration)

                if f1_dice > best_f1_dice:
                    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'best_f1_dice.pth'))
                    best_f1_dice = f1_dice
                    with open(os.path.join(args.ckpt_dir, f'best_f1_dice.txt'), 'a+') as f:
                        f.write(str(best_f1_dice) + '\n')
                if bg_dice > best_bg_dice:
                    torch.save(net.state_dict(), os.path.join(args.ckpt_dir, f'best_bg_dice.pth'))
                    best_bg_dice = bg_dice
                    with open(os.path.join(args.ckpt_dir, f'best_bg_dice.txt'), 'a+') as f:
                        f.write(str(best_bg_dice) + '\n')

        if args.task == 'reconstruction':
            r_loss = reconstruction(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=input_mod, writer=writer, trainer=trainer)
            if args.evaluate_only:
                exit()

        return

    # Stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_stats_handler.attach(evaluator)

    # Handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.iteration,
        log_dir=args.log_dir,
    )  # fetch global iteration number from trainer
    val_tensorboard_stats_handler.attach(evaluator)


    if not args.evaluate_only:
        train_epochs = args.epochs
        state = trainer.run(train_loader, train_epochs)
    else:

        run_validation(evaluator)
        exit()

        state = evaluator.run(val_loader)
    print(state)
