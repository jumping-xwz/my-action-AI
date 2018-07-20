import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(model_path):
    vgg19model = scipy.io.loadmat(model_path)
    mean = vgg19model['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = vgg19model['layers'][0]
    return weights, mean_pixel


def net_preloaded(weights, inputImg, pooling):
    net = {}
    current = inputImg
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]

        if kind == 'conv':
            Weight, Bias = weights[i][0][0][0][0]
            Weight = np.transpose(Weight, (1, 0, 2, 3))
            Bias = Bias.reshape(-1)
            conv = tf.nn.conv2d(current, tf.constant(Weight), strides=(1, 1, 1, 1), padding='SAME')
            current = tf.nn.bias_add(conv, Bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            if pooling == 'avg':
                current = tf.nn.avg_pool(current, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
            else:
                current = tf.nn.max_pool(current, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net


def preprocess(image, vgg_mean_pixel):
    return image - vgg_mean_pixel

def unprocess(image, vgg_mean_pixel):
    return image + vgg_mean_pixel