import pandas as pd
import os

# MONAI
from monai.utils import set_determinism
from monai.data import Dataset, CacheDataset, PersistentDataset, ThreadDataLoader
from monai.apps import DecathlonDataset
from monai.data import list_data_collate

# Utils
from utils.transforms import prepare_transforms
from utils.generate_mip import generate_mip
from utils.utils import prepare_input_mod

def process_MIPs(train_val_MIP, dim):
    path_MIP_scans_train_val_dim = sorted([os.path.join(f'{train_val_MIP}/mip_{dim}/', el) for el in sorted(os.listdir(f'{train_val_MIP}/mip_{dim}/'))])
    return path_MIP_scans_train_val_dim

def read_fns(in_dir=None, args=None):

    # AutoPET path for in_dir should point to FDG-PET-CT-Lesions/'
    set_determinism(seed=0)
    print(f"Input directory: {in_dir}")

    def class_label(ct_path, neg_paths):
        if ct_path[:-12] in neg_paths: # len(CTres.nii.gz) == 12
            return 0.0
        return 1.0

    if args.dataset == 'AutoPET':
        df_cv = pd.read_csv('data/autopet_5folds_augmented.csv')
        #df_cv['study_location'] = df_cv['study_location'].str.replace('/projects/datashare/tio/autopet/FDG-PET-CT-Lesions/', in_dir)
    else:
        df_cv = pd.read_csv('data/acrin_5folds_augmented.csv')

    if not args.with_negatives:
        df_cv_train = df_cv[(df_cv['kfold'] != args.fold) & (df_cv['diagnosis'] != 'NEGATIVE')]
        df_cv_val = df_cv[(df_cv['kfold'] == args.fold) & (df_cv['diagnosis'] != 'NEGATIVE')]
    else:
        df_cv_train = df_cv[df_cv['kfold'] != args.fold]
        df_cv_val = df_cv[df_cv['kfold'] == args.fold]
    negative_fns = df_cv[df_cv['diagnosis'] == 'NEGATIVE']['study_location'].to_numpy()


    path_train_files = df_cv_train['study_location'].to_numpy()
    path_val_files = df_cv_val['study_location'].to_numpy()

    # Train
    path_CT_scans_train = [os.path.join(el, 'CTres.nii.gz') for el in path_train_files]
    path_PET_scans_train = [os.path.join(el, 'SUV.nii.gz') for el in path_train_files]
    path_SEG_scans_train = [os.path.join(el, 'SEG.nii.gz') for el in path_train_files]

    # Val
    path_CT_scans_val = [os.path.join(el, 'CTres.nii.gz') for el in path_val_files]
    path_PET_scans_val = [os.path.join(el, 'SUV.nii.gz') for el in path_val_files]
    path_SEG_scans_val = [os.path.join(el, 'SEG.nii.gz') for el in path_val_files]


    # Tumor / No-tumor binary classes (0.0 or 1.0)
    train_class_labels = [class_label(el, negative_fns) for el in path_CT_scans_train]
    val_class_labels = [class_label(el, negative_fns) for el in path_CT_scans_val]

    # Maximum Intensity Projections (MIPs) used for classification
    train_MIP = f'./data/MIP/fold_{args.fold}/train_data'
    val_MIP = f'./data/MIP/fold_{args.fold}/val_data'

    seg_train_MIP = train_MIP
    seg_val_MIP = val_MIP

    if args.debrain:
        print('Using the MIPs with removed brains...')
        train_MIP = f'./data/MIP/threshold_exp/de-brain/fold_{args.fold}/train_data/'
        val_MIP = f'./data/MIP/threshold_exp/de-brain/fold_{args.fold}/val_data/'

    path_MIP_scans_train_x = process_MIPs(train_MIP, 'x')
    path_MIP_scans_val_x = process_MIPs(val_MIP, 'x')
    path_MIP_scans_train_y = process_MIPs(train_MIP, 'y')
    path_MIP_scans_val_y = process_MIPs(val_MIP, 'y')
    path_MIP_scans_train_z = process_MIPs(train_MIP, 'z')
    path_MIP_scans_val_z = process_MIPs(val_MIP, 'z')

    # 2D Seg-MIPs are only pre-computed for fold 0 (for now)
    if not args.generate_mip:
        path_MIP_segs_train_x = process_MIPs(seg_train_MIP + '/pngs/seg/', 'x')
        path_MIP_segs_val_x = process_MIPs(seg_val_MIP + '/pngs/seg/', 'x')
        path_MIP_segs_train_y = process_MIPs(seg_train_MIP + '/pngs/seg/', 'y')
        path_MIP_segs_val_y = process_MIPs(seg_val_MIP + '/pngs/seg/', 'y')
        path_MIP_segs_train_z = process_MIPs(seg_train_MIP + '/pngs/seg/', 'z')
        path_MIP_segs_val_z = process_MIPs(seg_val_MIP + '/pngs/seg/', 'z')


    if args.generate_mip: # Placeholders during data generation
        path_MIP_scans_train_x = ['dummy'] * len(train_class_labels)
        path_MIP_scans_train_y = ['dummy'] * len(train_class_labels)
        path_MIP_scans_train_z = ['dummy'] * len(train_class_labels)
        path_MIP_scans_val_x = ['dummy'] * len(val_class_labels)
        path_MIP_scans_val_y = ['dummy'] * len(val_class_labels)
        path_MIP_scans_val_z = ['dummy'] * len(val_class_labels)
        path_MIP_segs_train_x = ['dummy'] * len(train_class_labels)
        path_MIP_segs_train_y = ['dummy'] * len(train_class_labels)
        path_MIP_segs_train_z = ['dummy'] * len(train_class_labels)
        path_MIP_segs_val_x = ['dummy'] * len(val_class_labels)
        path_MIP_segs_val_y = ['dummy'] * len(val_class_labels)
        path_MIP_segs_val_z = ['dummy'] * len(val_class_labels)



    train_files = [{"ct_vol": ct_name,
                    "pet_vol": pet_name,
                    "seg": label_name,
                    "class_label": class_label,
                    "mip_x": mip_x,
                    "mip_y": mip_y,
                    "mip_z": mip_z,
                    'mip_fn_x': mip_fn_x,
                    "mip_seg_x": mip_seg_x,
                    "mip_seg_y": mip_seg_y,
                    "mip_seg_z": mip_seg_z}
                    for ct_name,
                    pet_name,
                    label_name,
                    class_label,
                    mip_x,
                    mip_y,
                    mip_z,
                    mip_fn_x,
                    mip_seg_x,
                    mip_seg_y,
                    mip_seg_z in
                    zip(path_CT_scans_train,
                    path_PET_scans_train,
                    path_SEG_scans_train,
                    train_class_labels,
                    path_MIP_scans_train_x,
                    path_MIP_scans_train_y,
                    path_MIP_scans_train_z,
                    path_MIP_scans_train_x,
                    path_MIP_segs_train_x,
                    path_MIP_segs_train_y,
                    path_MIP_segs_train_z)]

    val_files = [{"ct_vol": ct_name,
                  "pet_vol": pet_name,
                  "seg": label_name,
                  "class_label": class_label,
                  "mip_x": mip_x,
                  "mip_y": mip_y,
                  "mip_z": mip_z,
                  'mip_fn_x': mip_fn_x,
                  "mip_seg_x": mip_seg_x,
                  "mip_seg_y": mip_seg_y,
                  "mip_seg_z": mip_seg_z}
                  for ct_name,
                  pet_name,
                  label_name,
                  class_label,
                  mip_x,
                  mip_y,
                  mip_z,
                  mip_fn_x,
                  mip_seg_x,
                  mip_seg_y,
                   mip_seg_z in
                  zip(path_CT_scans_val,
                  path_PET_scans_val,
                  path_SEG_scans_val,
                  val_class_labels,
                  path_MIP_scans_val_x,
                  path_MIP_scans_val_y,
                  path_MIP_scans_val_z,
                  path_MIP_scans_val_x,
                  path_MIP_segs_val_x,
                  path_MIP_segs_val_y,
                  path_MIP_segs_val_z)]


    if args.evaluate_only:
        train_files = train_files[:4] # Skin loading whole training dataset
        val_files = val_files

    if args.debug:
        train_files = train_files[:4]
        val_files = train_files[:4] # test overfitting with small number of samples

    print('Train - positives:', len([el for el in train_files if el['class_label'] == 1.0]), 'negatives:', len([el for el in train_files if el['class_label'] == 0.0]))
    print('Val - positives:', len([el for el in val_files if el['class_label'] == 1.0]), 'negatives:', len([el for el in val_files if el['class_label'] == 0.0]))

    if args.generate_mip:
        print("Generating MIPs (this might take a while)...")
        generate_mip(train_files, 'train', args)
        generate_mip(val_files, 'val', args)
        exit()


    print('Training samples:', len(train_files), ' Validation samples:', len(val_files))

    return train_files, val_files

def prepare_loaders(in_dir=None, pixdim=(2.0, 2.0, 3.0), a_min_ct=-100, a_max_ct=250, a_min_pet=0, a_max_pet=15, spatial_size=[400, 400, 128], cache=True, args=None):

    train_transforms, val_transforms = prepare_transforms(pixdim=pixdim, a_min_ct=a_min_ct, a_max_ct=a_max_ct, a_min_pet=a_min_pet, a_max_pet=a_max_pet, spatial_size=spatial_size, args=args)
    cache = not args.no_cache

    if args.dataset == 'BraTS':
        train_ds = DecathlonDataset(
            root_dir=in_dir,
            task="Task01_BrainTumour",
            transform=train_transforms,
            section="training",
            download=True,
            cache_rate=0.0,
            num_workers=4,
        )
        val_ds = DecathlonDataset(
            root_dir=in_dir,
            task="Task01_BrainTumour",
            transform=val_transforms,
            section="validation",
            download=False,
            cache_rate=0.0,
            num_workers=4,
        )
        print('Training samples:', len(train_ds), ' Validation samples:', len(val_ds))

    else: # AutoPET
        train_files, val_files = read_fns(in_dir=in_dir, args=args)
        if cache:
            if args.task == 'classification' or args.generate_mip: # No need for persistent storage
                train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, copy_cache=False)
                val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, copy_cache=False)
            else:
                input_mod = prepare_input_mod(args)
                task_cache_dir = f'./cache/{args.task}/{input_mod}/sliding_window_{args.sliding_window}/with_DA_{args.with_DA}'

                if args.task in ['transference', 'fission', 'fission_classification', 'reconstruction', 'alt_transference']:
                    task_cache_dir = task_cache_dir.replace(args.task, 'segmentation') # Segmentation and args.task should share the cache dir
                if args.debug:
                    task_cache_dir = os.path.join(task_cache_dir, 'debug')
                if args.with_negatives:
                    task_cache_dir = os.path.join(task_cache_dir, 'with_negatives')
                if args.dataset == 'ACRIN':
                    task_cache_dir = os.path.join(task_cache_dir, args.dataset)

                task_cache_dir = task_cache_dir if args.cache_dir is None else args.cache_dir
                print(f"Using cache directory: {task_cache_dir}")
                train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=os.path.join(task_cache_dir, 'train'))
                if args.debug:
                    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=os.path.join(task_cache_dir, 'train'))
                else:
                    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=os.path.join(task_cache_dir, 'val'))
        else:
            train_ds = Dataset(data=train_files, transform=train_transforms)
            val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = ThreadDataLoader(train_ds,
                              batch_size=args.batch_size,
                              num_workers=0, # must be 0 to avoid "unexpected exception"
                              collate_fn=list_data_collate,
                              )
    val_batch_size = 1 if args.sliding_window else args.batch_size
    val_loader = ThreadDataLoader(val_ds,
                              batch_size=val_batch_size,
                              num_workers=0,
                              collate_fn=list_data_collate,
                              #pin_memory=torch.cuda.is_available() # removed due to EnsureTyped transform
                              )

    return train_loader, val_loader
