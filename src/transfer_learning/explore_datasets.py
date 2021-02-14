import os
from shutil import copyfile
from tqdm import tqdm


def create_folder(path2folder):
    if os.path.isdir(path2folder) is False:
        os.mkdir(path2folder)


def obtain_lines_txt(path2file):
    with open(path2file) as f:
        lines = [x.strip() for x in f.readlines()]
        # lines = f.readlines()

    return lines


def split_mit67_train_test(path2store, path2trainfile, path2testfile):
    """

    Args:
        path2store (str):
        path2trainfile (str):
        path2testfile (str):
    """
    path2split = os.path.join(path2store, 'split')
    create_folder(path2split)

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

    # Copy test files
    print('Copy testing files to corresponding label folder...')
    for jj in tqdm(range(0, len(testfiles))):
        copyfile(os.path.join(path2images, testfiles[jj]), os.path.join(path2test_split, testfiles[jj]))


if __name__ == '__main__':
    home = os.getenv('HOME')

    path2store = os.path.join(home, 'repos/DCASE2021-Task1b/data/mit67/indoorCVPR_09')
    path2trainfile = os.path.join(home, 'repos/DCASE2021-Task1b/data/mit67/TrainImages.txt')
    path2testfile = os.path.join(home, 'repos/DCASE2021-Task1b/data/mit67/TestImages.txt')

    split_mit67_train_test(path2store, path2trainfile, path2testfile)
