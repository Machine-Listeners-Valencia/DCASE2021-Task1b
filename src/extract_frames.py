import cv2
import os


def get_images_from_video(path2video, path2store, second=1, extension='.jpg'):
    """
    Function that slides a video and saves image according to seconds period
    Args:

        path2video (str): video path to be splitted
        path2store (str): path to store the image file
        second (int): value indicating every how many seconds an image is created
        extension (str): image format to be stored
    """

    filename = path2video.split('/')[-1][:-4]  # getting video name without extension TODO: improve
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


if __name__ == '__main__':
    home = os.getenv('HOME')

    get_images_from_video(home + '/repos/DCASE2021-Task1b/dummy_wav/toxicidad.mp4',
                          home + '/repos/DCASE2021-Task1b/dummy_wav/images/')
