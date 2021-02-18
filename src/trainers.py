from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import os
from generators import AudioMixupGenerator
import logging
import time
from utils import convert_to_preferred_format


def image_trainer(model, path2data, callbacks=None):
    logger = logging.getLogger(__name__)

    train_gen = ImageDataGenerator(**config.image_train_gen_args)
    val_gen = ImageDataGenerator(**config.image_val_gen_args)

    train_generator = train_gen.flow_from_directory(
        os.path.join(path2data, 'train'),  # this is the target directory
        target_size=config.image_network_settings['input_shape'][0:2],  # all images will be resized
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=True)

    val_generator = val_gen.flow_from_directory(
        os.path.join(path2data, 'val'),  # this is the target directory
        target_size=config.image_network_settings['input_shape'][0:2],  # all images will be resized
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=True)

    # n_training_files = sum([len(files) for r, d, files in os.walk(path2data + '/train')])
    # n_val_files = sum([len(files) for r, d, files in os.walk(path2data + '/val')])

    # TODO: callbacks and use .fit without data generator
    # TODO: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # TODO: https://keras.io/api/preprocessing/image/
    # TODO: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

    logger.info('STARTING FITTING')

    start_time = time.time()

    model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks)

    fitting_time = (time.time() - start_time)
    fitting_time = convert_to_preferred_format(fitting_time)

    logger.debug('FITTING TIME: {}'.format(fitting_time))
    logger.info('TRAINING FINISHED SUCCESSFULLY')


def audio_trainer(model, path2features, callbacks=None):
    import h5py

    logger = logging.getLogger(__name__)
    logger.info('LOADING DATA')

    hf_train = h5py.File(path2features + 'train.h5')  # TODO
    x_train = hf_train['features']
    y_train = hf_train['labels']
    hf_train.close()

    hf_val = h5py.File(path2features + 'val.h5')
    x_val = hf_val['features']
    y_val = hf_val['labels']
    hf_val.close()

    logger.info('CREATING MIXUP GENERATOR')
    audio_gen = AudioMixupGenerator(x_train=x_train, y_train=y_train,
                                    alpha=config.audio_train_gen_args['alpha'])

    logger.info('STARTING FITTING')

    start_time = time.time()

    model.fit(audio_gen,
              epochs=config.epochs,
              validation_data=(x_val, y_val),
              callbacks=callbacks)

    fitting_time = (time.time() - start_time)
    fitting_time = convert_to_preferred_format(fitting_time)

    logger.debug('FITTING TIME: {}'.format(fitting_time))
    logger.info('TRAINING FINISHED SUCCESSFULLY')


if __name__ == '__main__':
    image_trainer()
