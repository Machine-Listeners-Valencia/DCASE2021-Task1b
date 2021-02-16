from time import gmtime, strftime
import os


def create_training_outputs_folder(path2store):
    folder_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    folder_path = os.path.join(path2store, folder_name)

    if os.path.isdir(folder_path) is False:
        os.mkdir(folder_path)

    return folder_path
