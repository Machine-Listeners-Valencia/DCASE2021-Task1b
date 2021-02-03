import image_networks
import audio_networks
import config


def construct_joint_network():
    image_model = image_networks.construct_keras_image_network(include_classification=False)
    audio_model = audio_networks.construct_asc_network_csse(**config.audio_network_settings)

    # TODO: merge models
    #  https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0


if __name__ == '__main__':
    construct_joint_network()
