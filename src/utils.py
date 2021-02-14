import os


def create_folder(path2folder):
    if os.path.isdir(path2folder) is False:
        os.mkdir(path2folder)


def obtain_lines_txt(path2file):
    with open(path2file) as f:
        lines = f.readlines()

    return lines
