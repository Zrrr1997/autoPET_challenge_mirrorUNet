# Mirror U-Net: Marrying Multimodal Fission with Multi-task Learning for Semantic Segmentation in Medical Imaging
This repository contains the implementation for Mirror U-Net for training and evaluation on the AutoPET dataset. 

## Four Mirror U-Net settings:
The four settings can be trained and evaluated using the `--task` argument. For example, for training the decision fusion (a) just add `--task segmentation` to the training script. 

```
python test_multimodal_fission.py --task segmentation 
```

Tasks (b)-(d) correspond to `transference`, `fission`, and `fission_classification`. 

![cvpr_two_columns_4_paradigms (2)](https://user-images.githubusercontent.com/40185241/216580053-c043e3d7-999f-419c-9754-41e0135808dd.png)

## Hyperparameters
**Loss**
The loss function can be set to either `Dice` or `Dice_CE` with the `--loss` flag. You can choose whether to include the gradient from the background error by adding the `--include_background` flag. 

**Learning Rate**
The learning rate is set, e.g., by `--lr 0.0001`. You can set a learning rate step decay with `--lr_step_size 250` to decrease the learning rate by 10 at the 250th epoch.

**Data Augmentations**
To enable data augmentations (Rotations, Scaling etc. (see `utils/transforms.py')), just append the `--with_DA` flag.

**Inference**
The default inference is full-volume inference, where the volumes are resized to a common resolution of `[400, 400, 128]` to fit into memory. This leads to distortion artifacts which can be solved by using the sliding window inference by simply appending the `--sliding_window` flag. 

## Logging and Model Saving
The `--log_dir [YOUR_LOG_DIR]` flag sets where to store the `Tensorboard` logs of the training as well as additionally visualized outputs. The `--ckpt_dir` sets the path where to store the model checkpoints. The frequency to evaluate and store models can be set by `--eval_every` and `--save_every` respectively.

**Load Model Weights** To load the weights of a train model use `--load_weights [MODEL_PATH].pt`.


## Dataset
**ROOT Directory** 
Specify the root directory where you have all the PET/CT nifti files by the `--in_dir [YOUR_DIR_ENDING_IN_FDG-PET-CT-Lesions/` flag.

**Include Healthy Patients**
To include the healthy patients (with empty GT-masks) just add `--with_negatives`. 

**Debugging** To debug any model on only 2 samples and spare time from data loading add the `--debug` flag.

## Other Flags
Run `python test_multimodal_fission.py -h`.


## Additional Features
We also support training binary classifiers to identify whether a tumor is present or not. The classifiers are trained on 2D Maximum Intensity Projections (MIPs) of the PET volumes and can be used by adding `--task classification` to the script. 




