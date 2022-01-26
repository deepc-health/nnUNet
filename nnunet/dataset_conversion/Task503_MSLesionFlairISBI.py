import os
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, maybe_mkdir_p
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
from shutil import copyfile

 

if __name__ == '__main__':

    # download dataset from https://www.kaggle.com/insaff/massachusetts-roads-dataset
    # extract the zip file, then set the following path according to your system:
    base = 'ISBI/Dataset'
    # this folder should have the training and testing subfolders
        # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    labels_dir_tr = join(base, 'Train', 'segments')
    images_dir_tr = join(base, 'Train', 'Images')
    training_cases = subfiles(labels_dir_tr, suffix='.nii.gz', join=False)

    # now do the same for the test set
    labels_dir_ts = join(base, 'Train', 'segments')
    images_dir_ts = join(base, 'Train', 'Images')
    testing_cases = subfiles(labels_dir_ts, suffix='.nii.gz', join=False)
    

    # now start the conversion to nnU-Net:
    task_name = 'Task503_MSLesionFlairISBI'
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    print('Training Cases:', len(training_cases))
    for t in training_cases:
        unique_name =  os.path.basename(t).split('.')[0]
        input_segmentation_file = join(labels_dir_tr, t)
        input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr, unique_name+'_0000.nii.gz')  
        output_seg_file = join(target_labelsTr, unique_name+'.nii.gz')  

        copyfile(input_image_file, output_image_file)
        copyfile(input_segmentation_file, output_seg_file)
        

    print('Testing Cases:', len(testing_cases))

    for ts in testing_cases[0:3]:
        unique_name =  os.path.basename(ts).split('.')[0]
        input_segmentation_file = join(labels_dir_ts, ts)
        input_image_file = join(images_dir_ts, ts)

        output_image_file = join(target_imagesTs, unique_name+'_0000.nii.gz')
        output_seg_file = join(target_labelsTs, unique_name+'.nii.gz')

        copyfile(input_image_file, output_image_file)
        copyfile(input_segmentation_file, output_seg_file)

    # finally we can call the utility for generating a dataset.json

    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('FLAIR',),
                          labels={0: 'background', 1: 'Lesion' }, dataset_name=task_name, license='hands off!', suffix='.nii.gz')

    """
    once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the 
    `nnUNet_plan_and_preprocess` command like this:
    
    > nnUNet_plan_and_preprocess -t 120 -pl3d None
    
    once that is completed, you can run the trainings as follows:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD
    
    (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    
    there is no need to run nnUNet_find_best_configuration because there is only one model to choose from.
    Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    `nnUNet_determine_postprocessing` command
    """
