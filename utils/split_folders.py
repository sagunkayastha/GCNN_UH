import os
import random
import argparse
import shutil
from tqdm import tqdm


random.seed(922022)


def Train_val_split(input_folder, output_folder, split_percent=90):
    """
    Splits given input folder to train and validation
    
    Parameters
    ----------
    input_folder : path
        input folder path to split
    output_folder : path
        output folder path
    split_percent : int, optional
        percentage for splitting, by default 90
    """
    os.makedirs(output_folder, exist_ok=True)
    output_folder_train = os.path.join(output_folder, "Train")
    output_folder_validation = os.path.join(output_folder, "Validation")
    
    os.makedirs(output_folder_train , exist_ok=True)
    os.makedirs(output_folder_validation, exist_ok=True)
    
    
    # Listing all files
    all_files = os.listdir(input_folder)
    k = int(len(all_files) * split_percent // 100)
    random.shuffle(all_files)
    training_list = all_files[:k]
    validation_list = all_files[k:]

    # copy training files
    for file in tqdm(training_list):
        source = os.path.join(input_folder, file)
        shutil.copy(source, output_folder_train )
        
    # copy validation files
    for file in tqdm(validation_list):
        source = os.path.join(input_folder, file)
        shutil.copy(source, output_folder_validation)
        
        
if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Split Folders')
    parser.add_argument('--input_folder', default=' ', type=str,
                    help='The input folder path')
    parser.add_argument('--output_folder', default='./data_flist/train_shuffled.flist', type=str,
                    help='The output folder.')
    parser.add_argument('--split_percent', default='90', type=int,
                    help='Split Percentage')
    
    args = parser.parse_args()
    Train_val_split(args.input_folder, args.output_folder, args.split_percent)