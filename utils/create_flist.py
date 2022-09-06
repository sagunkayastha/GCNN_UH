#!/usr/bin/python

import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/dataFs/skayasth/CelebAMask-HQ/training_data/', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data/train_shuffle.flist', type=str,
                    help='The output filename.')
parser.add_argument('--validation_filename', default='./data/validation_shuffle.flist', type=str,
                    help='The output filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to shuffle')

def create_flist():
    args = parser.parse_args()
    # get the list of directories
    dirs = os.listdir(args.folder_path)
    dirs_name_list = []

    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []

    cwd = os.getcwd()
    train_folder = os.listdir(os.path.join(args.folder_path, "Train"))
    validation_folder = os.listdir(os.path.join(args.folder_path, "Validation"))
    
    for training_item in train_folder:
        # if "jpg" not in training_item or:
        #     continue
        training_item = os.path.join(args.folder_path, training_item)
        training_file_names.append(training_item)

    for validation_item in validation_folder:
        validation_item = os.path.join(args.folder_path, validation_item)
        validation_file_names.append(validation_item)
    # print all file paths
    # for i in training_file_names:
    #     print(i)
    # for i in validation_file_names:
    #     print(i)

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)
    print("Written file is: ", args.validation_filename, ", is_shuffle: ", args.is_shuffled)

if __name__ == "__main__":

    create_flist()

    