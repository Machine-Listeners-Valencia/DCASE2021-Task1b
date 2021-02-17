import config
from training_callbacks import create_callbacks
from utils import create_training_outputs_folder, save_to_json
import os


def train(path2store):

    path2callbacks = create_training_outputs_folder(path2store)

    callbacks = create_callbacks(path2callbacks, **config.callbacks_settings)
    save_to_json(os.path.join(path2callbacks, 'callbacks.json'), config.callbacks_settings)

    if config.which_train == 'image':
        from trainers import image_trainer
        from image_networks import construct_keras_image_network
        model = construct_keras_image_network(include_classification=True,
                                              **config.image_network_settings)

        save_to_json(os.path.join(path2callbacks, 'image_network_settings.json'),
                     config.image_network_settings)
        save_to_json(save_to_json(os.path.join(path2callbacks, 'image_generator.json'),
                     config.image_train_gen_args))

        image_trainer(model, config.path2image_data, callbacks=callbacks)

    if config.which_train == 'audio':
        print('TODO')


if __name__ == '__main__':

    home = os.getenv('HOME')

    train(config.path2outputs)
