import os
import sys
import getopt
from glob import glob

# file_root(base directory) of imagenet train images
def generate_valid_from_train(args):
    file_root =  os.path.join(args.data, 'train')
    per_class = args.per_class
    save_file = args.filelist

    """ get & sort subdirectories for labeling """
    # get list of subdirectory (each subdirectory corresponds to a single class)
    list_subdir = os.listdir(file_root)
    # sorting list of strings in lexicographic order 
    list_subdir = sorted(list_subdir)

    """get random 50 images from each classes"""
    # open txt file
    f = open(save_file, 'w')
    # scan subdirectories
    for label in range(len(list_subdir)):
        subdir = list_subdir[label]
        print("working on subdir {} ({}/1000)".format(subdir, label+1))
        folder = os.path.join(file_root, subdir)
        files = [name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
        for i in range(per_class):
            file_name = '{}/{}'.format(subdir, files[i])
            new_line = "{} {}\n".format(file_name, label)
            f.write(new_line)
    # close txt file
    f.close()



