import os


def create_folder(path2folder):
    if os.path.isdir(path2folder) is False:
        os.mkdir(path2folder)
