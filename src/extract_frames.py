import cv2
import os


def get_images_from_video(path2video, path2store, second=1):
    """

    Args:
        path2video (str): video path to be splitted
        path2store (str): path to store the image file
        second (int): value indicating every how many seconds an image is created
    """

    # TODO: filename that can be mapped to the video, extension of the image (jpg or png)

    vidcap = cv2.VideoCapture(path2video)
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    while success:
        success, image = vidcap.read()
        # print('read a new frame:', success)
        if count % (second * fps) == 0:
            cv2.imwrite(path2store + 'frame%d.jpg' % count, image)
        #    print('successfully written 10th frame')
        count += 1


if __name__ == '__main__':

    home = os.getenv('HOME')

    get_images_from_video(home + '/repos/DCASE2021-Task1b/dummy_wav/toxicidad.mp4',
                          home + '/repos/DCASE2021-Task1b/dummy_wav/images/')
