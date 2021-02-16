import config
from training_callbacks import create_callbacks
from utils import create_training_outputs_folder


def train(path2store):
    # TODO: dict with settings

    path2callbacks = create_callbacks(path2store)

    callbacks = create_callbacks(path2callbacks, **config.callbacks_settings)

    if config.which_train == 'image':
        from trainers import image_trainer
        from image_networks import construct_keras_image_network
        model = construct_keras_image_network(include_classification=True,
                                              **config.image_network_settings)

        image_trainer(model, config.path2image_data, callbacks=callbacks)


if __name__ == '__main__':
    train()
