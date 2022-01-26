import torch
import importlib
import pkgutil
import pickle
from os.path import join
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import datetime

def dice_coef(y_true, y_pred):
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1.)

def gt_transform(img):
     img[img== 50]= 1
     img[img == 100] = 2
     img[img == 200] =3
     img[img == 180] =4
     img[img == 190] =5

     return img.astype(int)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for _ , modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr

def restore_model(pkl_file, checkpoint=None, train=False):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    search_in = "/home/ubuntu/nnUNet/nnunet/training/network_training"
    tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")
    trainer = tr(*init)


    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer

all_best_model_files = ["./model_best.model"]
trainer = restore_model("./model_best.model.pkl", "./model_best.model", False)
params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files][0]

print("starting preprocessing generator")

print("starting prediction...")
list_of_all_files = glob.glob('/home/ubuntu/nnUNet/JSRT/Dataset/Test/Images/*.png')
df = pd.DataFrame(columns=['Name', 'L. Lung', 'R. Lung', 'Heart', 'R. Clavicle', 'L. Clavicle', 'Avg. Dice'])
for preprocessed in list_of_all_files:
    # load the image
    data = plt.imread(preprocessed)
    data = np.expand_dims(data, [0,1])
    #d = :  np.ndarray 1x1024x1024
    print("predicting", os.path.basename(preprocessed))
    output_filename = os.path.basename(preprocessed)[:-4]+'.png'

    gt = plt.imread(preprocessed.replace("Images","segments"))
    
    trainer.load_checkpoint_ram(params, False)
    
    time_list = []
    for i in range(20):
        start = datetime.datetime.now()
        seg = trainer.predict_preprocessed_data_return_seg_and_softmax(
            data, do_mirroring=False, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=0.5, use_gaussian=True, all_in_gpu=False, mixed_precision=None)[0]
        end = datetime.datetime.now()-start
        time_list.append(end)
    
    time = [time.total_seconds() for time in time_list]
    print(np.mean(time), np.median(time), np.std(time))
    import pdb; pdb.set_trace()
    # calculate metrics
    metric_line = [output_filename] 
    transformed_gt = gt_transform((gt*255).astype(np.uint8))
    for label in [1, 2, 3, 4, 5]:
        metric_line.append(dice_coef(transformed_gt==label, seg[0]==label))

    metric_line.append(np.mean(metric_line[1:]))
    df.loc[len(df)] = metric_line
    # plot img    
    masked = np.ma.masked_where(seg == 0, seg)
    plt.imshow(data[0][0], 'gray', interpolation='none')
    plt.imshow(masked[0], 'jet', interpolation='none', alpha=0.3)
    plt.axis('off')
    plt.savefig(output_filename)

metric_line = ["avg score"]
for label in [1, 2, 3, 4, 5]:
    metric_line.append(np.mean(df[df.columns[label]]))
metric_line.append(np.mean(metric_line[1:]))
df.loc[len(df)] = metric_line
df.to_csv('trail.csv', index=False)
