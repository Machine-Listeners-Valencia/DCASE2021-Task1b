from ..utils import create_folder
import os


def split_mit67_train_test(path2store, path2trainfile, path2testfile):
    path2split = os.path.join(path2store, 'split')
    create_folder(path2split)

    # TODO: create folders from training and testing and copy files
