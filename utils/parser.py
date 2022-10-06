import argparse

def prepare_parser(parser):
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3,
                         help='Learning rate.')
    parser.add_argument('--lr_step_size', type=int, default=250,
                         help='Decrease learning rate every lr_step_size epochs.')
    parser.add_argument('--batch_size', type=int, default=4,
                         help='Batch size for data loaders.')
    parser.add_argument('--epochs', type=int, default=400,
                         help='Training epochs.')
    parser.add_argument('--include_background', default=False, action='store_true',
                         help='Include error gradients from background during training.')
    parser.add_argument('--loss', type=str, default='DiceCE',
                         help='Loss function for segmentation training [Dice, DiceCE].')
    parser.add_argument('--with_DA', default=False, action='store_true',
                         help='Apply data augmentation during segmentation training.')

    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs/test',
                         help='Logs directory.')
    parser.add_argument('--eval_every', type=int, default=200,
                         help='Number of iterations to evaluate model.')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/test',
                         help='Checkpoint directory.')
    parser.add_argument('--save_every', type=int, default=20,
                         help='Number of epochs to save a model.')



    # Configuration
    parser.add_argument('--gpu', type=int, default=0,
                         help='GPU device index.')
    parser.add_argument('--debug', default=False, action='store_true',
                         help='Debug with one sample for training and validation.')
    parser.add_argument('--sliding_window', default=False, action='store_true',
                         help='Use sliding window inference.')
    parser.add_argument('--save_eval_img', default=False, action='store_true',
                         help='Save an image during evaluation.')
    parser.add_argument('--save_nifti', default=False, action='store_true',
                         help='Save nifti files of the output and ground truth.')

    # Dataset
    parser.add_argument('--with_negatives', default=False, action='store_true',
                         help='Include data samples without any tumor (empty label).')
    parser.add_argument('--fold', type=int, default=0,
                         help='Cross-validation fold to evaluate on: [0, 1, 2, 3, 4, 5].')
    parser.add_argument('--single_mod', type=str, default=None,
                         help='Training/Evaluating on single modality, e.g. pet_vol or ct_vol.')
    parser.add_argument('--evaluate_only', default=False, action='store_true',
                         help='Only evaluate without training.')
    parser.add_argument('--proj_dim', type=str, default=None,
                         help='Dimension on which to do the MIP projection: [x, y, z, all]')
    parser.add_argument('--debrain', default=False, action='store_true',
                         help='Use the de-brained MIPs.')
    parser.add_argument('--no_cache', default=False, action='store_true',
                         help='Toggle using CacheDataset/PersistentDataset.')

    # evaluation
    parser.add_argument('--logit_fusion', default=False, action='store_true',
                         help='Fusion by averaging the logits.')
    parser.add_argument('--decision_fusion', default=False, action='store_true',
                         help='Fusion by (weak) averaging the predictions.')
    parser.add_argument('--separate_outputs', default=False, action='store_true',
                         help='Return separate outputs for the two Mirror UNet arms.')



    # Model
    parser.add_argument('--load_weights', type=str, default=None,
                         help='Load model from this directory.')
    parser.add_argument('--load_weights_second_model', type=str, default=None,
                         help='Load second model from this directory.')
    parser.add_argument('--load_best_val_weights', type=str, default=None,
                         help='Load best validation model from the given directory.')
    parser.add_argument('--load_keyword', type=str, default=None,
                             help='Keyword to search for in the weights file.')
    parser.add_argument('--task', type=str, default='segmentation',
                         help='Training task for the model: [segmentation, reconstruction, classification, segmentation_classification, transference]')
    parser.add_argument('--early_fusion', default=False, action='store_true',
                         help='Train UNet with early fusion (channel concatenation).')
    parser.add_argument('--mask_attention', default=False, action='store_true',
                         help='Concatenate the distribution of tumor locations to the input channel (mask_attention).')
    parser.add_argument('--class_backbone', type=str, default='ResNet',
                         help='Classification backbone: [ResNet, EfficientNet, CoAtNet, Ensemble].')
    parser.add_argument('--resnet_version', type=str, default='resnet18',
                         help='Which ResNet version to use for the classification: [resnet18, resnet50, resnet101]')
    parser.add_argument('--resnet_dropout', default=False, action='store_true',
                         help='Apply dropout to ResNet"s penultimate layer.')
    parser.add_argument('--efficientnet_version', type=str, default='b0',
                         help='EfficientNet version: [b0, b4, widese_b0, widese_b4].')
    parser.add_argument('--coatnet_version', type=str, default='0',
                         help='CoAtNet version: [0, 1, 2, 3, 4].')
    parser.add_argument('--mirror_th', type=float, default=0.1,
                        help='Weight for CT-modalitiy in the late fusion for mirror-UNet (experiment 1).')
    parser.add_argument('--learnable_th', default=False, action='store_true',
                        help='Make the fusion threshold learnable for mirror-UNet in the late fusion for mirror-UNet (experiment 1).')
    parser.add_argument('--lambda_rec', type=float, default=1e-3,
                        help='Weight for the reconstruction loss in the transference (experiment 2).')
    parser.add_argument('--lambda_seg', type=float, default=0.5,
                        help='Weight for the segmentation loss in the transference (experiment 2).')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='True if loading pre-trained weights for initialization (required for tranference pre-initialization).')


    # Utils
    parser.add_argument('--generate_mip', default=False, action='store_true',
                             help='Generate maximum intensity projections (MIPs).')
    parser.add_argument('--generate_gradcam', default=False, action='store_true',
                         help='Generate GradCAM during evaluation.')
    parser.add_argument('--gradcam_start_index', type=int, default=0,
                         help='GradCAM stard index.')

    return parser
