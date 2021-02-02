from keras.preprocessing.image import ImageDataGenerator
import config
import os


def image_trainer(model, path2data):
    train_gen = ImageDataGenerator(**config.image_train_gen_args)
    val_gen = ImageDataGenerator(**config.image_val_gen_args)

    train_generator = train_gen.flow_from_directory(
        path2data + '/train',  # this is the target directory
        target_size=config.image_network_setting['target_size'],  # all images will be resized to 150x150
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=True)

    val_generator = val_gen.flow_from_directory(
        path2data + '/val',  # this is the target directory
        target_size=config.image_network_setting['target_size'],  # all images will be resized to 150x150
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=True)

    n_training_files = sum([len(files) for r, d, files in os.walk(path2data + '/train')])
    n_val_files = sum([len(files) for r, d, files in os.walk(path2data + '/val')])

    # TODO: callbacks
    model.fit_generator(
        train_generator,
        steps_per_epoch=n_training_files // config.batch_size,
        epochs=config.epochs,
        validation_data=val_generator,
        validation_steps=n_val_files // config.batch_size)


if __name__ == '__main__':
    image_trainer()