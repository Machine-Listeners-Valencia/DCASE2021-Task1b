import config
from training_callbacks import create_callbacks
from utils import create_training_outputs_folder, save_to_json, create_logger
import os
import multiprocessing

path2callbacks = create_training_outputs_folder(config.path2outputs)

logger = create_logger(os.path.join(path2callbacks, 'logger.log'), level='INFO')


def train(path2callbacks):
    logger.info('STARTING TRAINING')
    logger.info('TRAINING DOMAIN: {}'.format(config.which_train))
    logger.info('ALL TRAINING INFORMATION WILL BE STORED IN: {}'.format(path2callbacks))
    logger.info('CREATING CALLBACKS')
    callbacks = create_callbacks(path2callbacks, **config.callbacks_settings)
    save_to_json(os.path.join(path2callbacks, 'callbacks.json'), config.callbacks_settings)
    logger.info('CALLBACK CONFIGURATION CAN BE CHECKED IN: {}'.format(os.path.join(path2callbacks, 'callbacks.json')))

    if config.which_train == 'image':
        from trainers import image_trainer
        from image_networks import construct_keras_image_network
        logger.info('CONSTRUCTING IMAGE CLASSIFICATION NETWORK')
        model = construct_keras_image_network(include_classification=True,
                                              **config.image_network_settings)

        save_to_json(os.path.join(path2callbacks, 'image_network_settings.json'),
                     config.image_network_settings)
        logger.info(
            'IMAGE NETWORK CONFIGURATION CAN BE CHECKED IN: {}'.format(os.path.join(path2callbacks,
                                                                                    'image_network_settings.json')))
        save_to_json(os.path.join(path2callbacks, 'image_generator.json'),
                     config.image_train_gen_args)

        logger.info(
            'IMAGE TRAINING CONFIGURATION CAN BE CHECKED IN: {}'.format(os.path.join(path2callbacks,
                                                                                     'image_generator.json')))

        logger.info('FITTING IMAGE NETWORK ARCHITECTURE: {}'.format(config.image_network_settings['net']))
        image_trainer(model, config.path2image_data, callbacks=callbacks)

    if config.which_train == 'audio':
        from trainers import audio_trainer
        from audio_networks import construct_asc_network_csse
        logger.info('CONSTRUCTING AUDIO CLASSIFICATION NETWORK')
        model = construct_asc_network_csse(include_classification=True,
                                           **config.audio_network_settings)
        save_to_json(os.path.join(path2callbacks, 'audio_network_settings.json'),
                     config.audio_network_settings)
        logger.info(
            'AUDIO NETWORK CONFIGURATION CAN BE CHECKED IN: {}'.format(os.path.join(path2callbacks,
                                                                                    'audio_network_settings.json')))

        save_to_json(os.path.join(path2callbacks, 'audio_generator.json'),
                     config.audio_train_gen_args)

        logger.info(
            'AUDIO TRAINING CONFIGURATION CAN BE CHECKED IN: {}'.format(os.path.join(path2callbacks,
                                                                                     'audio_generator.json')))

        logger.info('FITTING AUDIO NETWORK ARCHITECTURE DCASE2020')
        audio_trainer(model, config.path2audio_data, callbacks=callbacks)


if __name__ == '__main__':
    #path2callbacks = create_training_outputs_folder(config.path2outputs)

    process_training = multiprocessing.Process(target=train, args=(path2callbacks,))
    process_training.start()
    process_training.join()

    #train(path2callbacks)
