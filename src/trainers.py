from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import os
from generators import AudioMixupGenerator


def image_trainer(model, path2data, callbacks=None):
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
    model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks)


def audio_trainer(model, path2features, callbacks=None):
    import h5py

    hf_train = h5py.File(path2features + 'train.h5')  # TODO
    x_train = hf_train['features']
    y_train = hf_train['labels']
    hf_train.close()

    hf_val = h5py.File(path2features + 'val.h5')
    x_val = hf_val['features']
    y_val = hf_val['labels']
    hf_val.close()

    audio_gen = AudioMixupGenerator(x_train=x_train, y_train=y_train,
                                    alpha=config.audio_train_gen_args['alpha'])

    model.fit(audio_gen,
              epochs=config.epochs,
              validation_data=(x_val, y_val),
              callbacks=callbacks)


if __name__ == '__main__':
    image_trainer()
