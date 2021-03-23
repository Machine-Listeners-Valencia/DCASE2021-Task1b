import cv2
import os
import pandas as pd
from tqdm import tqdm


def get_images_from_video(path2video, path2store, second=1, extension='.jpg'):
    """
    Function that slides a video and saves image according to seconds period
    Args:

        path2video (str): video path to be splitted
        path2store (str): path to store the image file
        second (int): value indicating every how many seconds an image is created
        extension (str): image format to be stored
    """

    # filename = path2video.split('/')[-1][:-4]  # getting video name without extension TODO: improve
    filename = os.path.splitext(os.path.basename(path2video))[0]
    vidcap = cv2.VideoCapture(path2video)
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    while success:
        success, image = vidcap.read()
        # print('read a new frame:', success)
        if count % (second * fps) == 0:
            cv2.imwrite(path2store + filename + '_frame{}'.format(count) + extension, image)
        #    print('successfully written 10th frame')
        count += 1


def extract_video_images(path2csv, mode='train', seconds=10):
    dataframe = pd.read_csv(path2csv, sep='\t')

    video_files = dataframe['filename_video'].tolist()
    video_labels = dataframe['scene_label'].tolist()

    unique_video_labels = list(set(video_labels))

    if os.path.isdir('../data/audiovisual/images/') is False:
        os.mkdir('../data/audiovisual/images/')

    if os.path.isdir('../data/audiovisual/images/{}'.format(mode)) is False:
        os.mkdir('../data/audiovisual/images/{}'.format(mode))

    for i in range(0, len(unique_video_labels)):

        if os.path.isdir('../data/audiovisual/images/{}/{}'.format(mode,
                                                                   unique_video_labels[i])) is False:
            os.mkdir('../data/audiovisual/images/{}/{}'.format(mode, unique_video_labels[i]))

    for i in tqdm(range(0, len(video_files))):
        path2store = '../data/audiovisual/images/{}/{}/'.format(mode, video_labels[i])
        get_images_from_video('../data/audiovisual/' + video_files[i], path2store, second=seconds, extension='.jpg')


if __name__ == '__main__':
    home = os.getenv('HOME')

    extract_video_images('../data/audiovisual/TAU-urban-audio-visual-scenes-2021-development.meta/'
                         'evaluation_setup/fold1_train.csv', mode='train')

    extract_video_images('../data/audiovisual/TAU-urban-audio-visual-scenes-2021-development.meta/'
                         'evaluation_setup/fold1_evaluate.csv', mode='evaluate')
