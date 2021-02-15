import tensorflow
from tensorflow.keras.applications import Xception, InceptionResNetV2, InceptionV3
import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import config


def construct_efficientnet(type, input_shape, verbose=True):
    if type == 0:
        primary_model = efn.EfficientNetB0(weights='imagenet')
    elif type == 1:
        primary_model = efn.EfficientNetB1(weights='imagenet')
    elif type == 2:
        primary_model = efn.EfficientNetB2(weights='imagenet')
    elif type == 3:
        primary_model = efn.EfficientNetB3(weights='imagenet')
    elif type == 4:
        primary_model = efn.EfficientNetB4(weights='imagenet')
    elif type == 5:
        primary_model = efn.EfficientNetB5(weights='imagenet')
    elif type == 6:
        primary_model = efn.EfficientNetB6(weights='imagenet')
    elif type == 7:
        primary_model = efn.EfficientNetB7(weights='imagenet')

    # TODO: change input shape
    # TODO: m  = efn.EfficientNetB0(weights='imagenet', input_shape = (100,100,3), include_top=False)
    # primary_model.layers.pop(0)  # removing input layer
    # primary_model.layers.pop(-1)

    model = Model(inputs=primary_model.input, outputs=primary_model.layers[-2].output)

    if verbose:
        print(model.summary())

    # inp = Input(shape=input_shape)  # creating layer with specific input shape

    # out = primary_model(inp)

    # model = Model(inputs=inp, outputs=out)

    # if verbose:
    #     print(model.summary())

    return model


def construct_keras_image_network(include_classification=True, **parameters):
    """
    Constructs a keras model with pretrained weights and top layers according to the classification problem
    Args:
        include_classification (bool): include classification layer
        **parameters: setting use to construct the network
            net (str): pretrained network type (xception, inception_resnet_v2 or inception_v3)
            include_top (bool): if including classification layer, typically set to False
            pooling (str): class of pooling before Dense layers: 'max' or 'avg'
            input_shape (tuple): size of the RBG images, typically (224, 224, 3)
            trainable (bool): if pretrained network layers can be optimized
            verbose (bool): if summary is showed

    Returns:
        keras model according to the type and the Dense layers at the top
    """

    net = parameters['net']
    input_shape = parameters['input_shape']
    include_top = parameters['include_top']
    pooling = parameters['pooling']
    trainable = parameters['trainable']
    verbose = parameters['verbose']

    if net is 'xception':
        base_model = Xception(input_shape=input_shape, include_top=include_top, pooling=pooling)

    elif net is 'inception_resnet_v2':
        base_model = InceptionResNetV2(input_shape=input_shape, include_top=include_top, pooling=pooling)

    elif net is 'inception_v3':
        base_model = InceptionV3(input_shape=input_shape, include_top=include_top, pooling=pooling)

    elif 'efficient' in net:
        base_model = construct_efficientnet(int(net.split('-')[1]), input_shape)

    for layer in base_model.layers:
        layer.trainable = trainable

    if include_classification:
        outputs = Dense(units=config.n_classes, activation='softmax')(base_model.layers[-1].output)
    else:
        outputs = base_model.layers[-1].output

    model = Model(inputs=base_model.inputs, outputs=outputs)

    if verbose:
        print(model.summary())

    return model


if __name__ == '__main__':
    model = construct_keras_image_network(include_classification=True,
                                          **config.image_network_settings)
