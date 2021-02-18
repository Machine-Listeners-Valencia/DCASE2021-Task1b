from time import gmtime, strftime
import os
import json


def create_training_outputs_folder(path2store):
    folder_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    folder_path = os.path.join(path2store, folder_name)

    if os.path.isdir(folder_path) is False:
        os.mkdir(folder_path)

    return folder_path


def save_to_json(json_name, data):
    with open(json_name, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)


def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    # print("seconds value in hours:",hour)
    # print("seconds value in minutes:",min)
    return "%02d:%02d:%02d" % (hour, min, sec)
