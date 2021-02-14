from ..utils import create_folder, obtain_lines_txt
import os
from shutil import copyfile
from tqdm import tqdm


def split_mit67_train_test(path2store, path2trainfile, path2testfile):
    path2split = os.path.join(path2store, 'split')
    create_folder(path2split)

    # TODO: create folders from training and testing and copy files
    trainfiles = obtain_lines_txt(path2trainfile)
    testfiles = obtain_lines_txt(path2testfile)

    path2images = os.path.join(path2store, 'Images')

    # obtaining classes/labels
    labels = [item.split('/')[0] for item in trainfiles]
    unique_labels = list(set(labels))

    path2train_split = os.path.join(path2split, 'train')
    path2test_split = os.path.join(path2split, 'test')

    create_folder(path2train_split)
    create_folder(path2test_split)

    # Create labels folders
    print('Creating labels folders...')
    for i in tqdm(range(0, len(unique_labels))):
        path2train_split_aux = os.path.join(path2train_split, unique_labels[i])
        path2test_split_aux = os.path.join(path2test_split, unique_labels[i])
        create_folder(path2train_split_aux)
        create_folder(path2test_split_aux)

    # Copy train files
    print('Copy training files to corresponding label folder...')
    for j in tqdm(range(0, len(trainfiles))):
        copyfile(os.path.join(path2images, trainfiles[j]), os.path.join(path2train_split, trainfiles[j]))

    print('Copy training files to corresponding label folder...')
    # Copy test files
    for jj in tqdm(range(0, len(testfiles))):
        copyfile(os.path.join(path2images, testfiles[jj]), os.path.join(path2test_split, testfiles[jj]))



