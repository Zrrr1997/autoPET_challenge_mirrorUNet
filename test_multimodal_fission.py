from tqdm import tqdm
import numpy as np
import os
import sys
import json
import logging
import argparse

# Torch
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchviz import make_dot



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
import monai

from monai.data import list_data_collate, decollate_batch

from monai.metrics import compute_meandice
from monai.handlers import (
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    LrScheduleHandler,
)

# Utils
from utils.data_utils import prepare_loaders
from utils.parser import prepare_parser
from utils.models import prepare_model
from utils.utils import *

# Eval
from utils.eval import *


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Use all cores
    print(f"CPU Count: {os.cpu_count()}")
    torch.set_num_threads(os.cpu_count())
    print(f"Num threads: {torch.get_num_threads()}")

    parser = argparse.ArgumentParser(description='AutoPET codebase implementation.')
    parser = prepare_parser(parser)
    args = parser.parse_args()
    print('--------\n')
    print(args, '\n')
    print('--------')



    out_channels = prepare_out_channels(args)
    input_mod = prepare_input_mod(args)


    best_f1_dice, best_bg_dice = 0, 0


    # Create Model, Loss, and Optimizer
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")
    net, net_2 = prepare_model(device=device, out_channels=out_channels, args=args)

    #summary(net, input_size=(args.batch_size, 2, 400, 400, 128))


    # Data configurations
    spatial_size = [224, 224, 128] if (args.class_backbone == 'CoAtNet' and args.task == 'classification') else [400, 400, 128]
    train_loader, val_loader, train_files, val_files = prepare_loaders(spatial_size=spatial_size, args=args)



    writer = SummaryWriter(log_dir = args.log_dir)

    # Write the current configuration to file
    with open(os.path.join(args.log_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    check_data = first(val_loader)
    print('Input shape check:', check_data[input_mod].shape)

    #yhat = net(check_data['ct_pet_vol'].to(device))
    #make_dot(yhat, params=dict(list(net.named_parameters()))).render(f"mirrorUNet_graph_level_{args.level}_depth_{args.depth}", format="png")
    #exit()


    check_data_shape(val_loader, input_mod, args)

    # Hyperparameters
    loss = prepare_loss(args)
    lr = args.lr

    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-5)

    attention = None
    if args.mask_attention:
        attention = prepare_attention(args)
        print('Attention shape:', attention.shape)

    def f_class_label(class_label, batch):
        if class_label != 0:
            class_label = torch.ones(batch.shape[1:])
        else:
            class_label = torch.zeros(batch.shape[1:])
        return class_label

    def prepare_batch(batch, device=None, non_blocking=False, task=args.task):
        if not args.mask_attention:
            inp = batch[input_mod]
        else:
            inp = torch.cat((torch.cat(batch[input_mod].shape[0] * [attention]), batch[input_mod]), dim=1) # dim=1 is the channel

        if task == 'segmentation' or task == 'segmentation_classification':
            return _prepare_batch((inp, batch["seg"]), device, non_blocking)
        elif task == 'reconstruction':
            return _prepare_batch((inp, batch[input_mod]), device, non_blocking)
        elif task == 'classification' and not args.mask_attention and args.sliding_window: # classification without mask, with sliding window inference
            seg = batch[f"mip_seg_{args.proj_dim}"]
            label = []
            for el in seg:
                label.append((torch.sum(el) > 0) * 1.0)
            label = torch.Tensor(label)
            return _prepare_batch((inp, label.unsqueeze(dim=1).float()), device, non_blocking)
        elif task == 'classification' and not args.mask_attention and args.proj_dim is None: # classification without mask, with 3 channels
            return _prepare_batch((inp, batch["class_label"].unsqueeze(dim=1).float()), device, non_blocking)
        elif task == 'classification' and args.proj_dim is not None: # classification with mask, with 1 channel
            return _prepare_batch((inp, batch["class_label"].unsqueeze(dim=1).float()), device, non_blocking)
        elif task == 'transference':

            ct_vol = inp[:, :1]

            return _prepare_batch((inp, torch.cat([ct_vol, batch['seg']], dim=1)), device, non_blocking)
        elif task == 'co-learning':
            ct_vol = inp[:, :1]

            class_labels = batch['class_label'] # TODO: Maybe this is inefficient and redundant
            class_label = torch.stack([f_class_label(el, ct_vol) for el in class_labels]).to(device)


            return _prepare_batch((inp, torch.cat([ct_vol, batch['seg'], class_label], dim=1)), device, non_blocking)
        else:
            print("[ERROR]: No such task exists...")
            exit()



    # Metric to evaluate whether to save the "best" model or not
    def default_score_fn(engine):
        if args.task == 'classification':
            score = engine.state.metrics['Accuracy']
        elif args.task == 'reconstruction':
            score = engine.state.metrics['MSE']
        else: # Segmentation, Transference, Co-Learning
            score = engine.state.metrics['Mean_Dice']
        return score
    def default_score_fn_F1(engine):
        return engine.state.metrics['Mean_Dice_F1'] # More representative for tumors


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
    if args.task in ['classification', 'segmentation', 'transference', 'reconstruction', 'co-learning']:
        checkpoint_handler_best_val = ModelCheckpoint(
            args.ckpt_dir, "net_best_val", n_saved=1, require_empty=False, score_function=default_score_fn
        )
        if args.task in ['segmentation', 'transference', 'co-learning']:
            checkpoint_handler_best_val_F1 = ModelCheckpoint(
                args.ckpt_dir, "net_best_val_F1", n_saved=1, require_empty=False, score_function=default_score_fn_F1
            )
            evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler_best_val_F1, {'net': net, })

        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler_best_val, {'net': net, })


    @trainer.on(Events.ITERATION_COMPLETED(every=validation_every_n_iters))
    def run_validation(engine):
        global best_f1_dice, best_bg_dice

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
        if args.task=='segmentation':
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
        ##         CO-LEARNING             ##
        #####################################
        if args.task=='co-learning':
            f1_dice, bg_dice = co_learning(args, val_loader, net, evaluator, post_pred, post_label, device, input_mod=input_mod, writer=writer, trainer=trainer)
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
        state = evaluator.run(val_loader)
    print(state)
