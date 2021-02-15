import config
from training_callbacks import create_callbacks


def train():
    # TODO: create folder and dict with settings
    callbacks = create_callbacks('jelou!', **config.callbacks_settings)

    if config.which_train == 'image':
        from trainers import image_trainer
        from image_networks import construct_keras_image_network
        model = construct_keras_image_network(include_classification=True,
                                              **config.image_network_settings)

        image_trainer(model, config.path2image_data, callbacks=callbacks)


if __name__ == '__main__':
    train()
