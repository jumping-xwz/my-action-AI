import numpy as np
import scipy.misc
import imageio
from PIL import Image
import vgg
import tensorflow as tf
from sys import stderr
from functools import reduce
import time

###########################################
# Gobal Const and fine tuning params define
###########################################
STYLE_LAYERS = ('relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYERS = ('relu4_2', 'relu5_2')

STYLE_SCALE = 1.0
INITIAL_NOISEBLEND = 1.0
PRESERVE_COLORS = 0.8

ITERATIONS = 1000
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1

STYLE_WEIGHT = 5e2
STYLE_LAYER_WEIGHT_EXP = 1
STYLE_BLEND_WEIGHT = 1.0

TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
POOLING = 'max' # use max-pooling.

PRINT_ITERATIONS = 20   # every 20 iterations to save inference pic.
CHECKPOINT_ITERATIONS = 20  # every 20 iterations to print loss message.

###########################
#   Tool func
###########################
def imgRead(path):
    img = imageio.imread(path).astype(np.float)
    return img


def imgSave(path, image):
    img = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


#########################################
#   inference image
#########################################
def inferenceImg(network, initial_img, initial_noiseblend,
                 content, style, preserve_colors, iterations,
                 content_weight, content_weight_blend,
                 style_weight, style_layer_weight_exp, style_blend_weight,
                 tv_weight, learning_rate, beta1, beta2, epsilon, pooling,
                 print_iterations, checkpoint_iterations
                 ):

    content_shape = (1,) + content.shape
    style_shape = (1,) + style.shape

    content_features = {}
    style_features = {}

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight = layer_weight * style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum = layer_weights_sum + style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = style_layers_weights[style_layer] / layer_weights_sum

    # compute content features in feedforward mode
    g1 = tf.Graph()
    with g1.as_default(), g1.device('/cpu:0'), tf.Session() as sess:
        contentImg = tf.placeholder('float', shape=content_shape)
        net = vgg.net_preloaded(vgg_weights, contentImg, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={contentImg: content_pre})

    # compute style features in feedforward mode
    g2 = tf.Graph()
    with g2.as_default(), g2.device('/cpu:0'), tf.Session() as sess:
        styleImg = tf.placeholder('float', shape=style_shape)
        net = vgg.net_preloaded(vgg_weights, styleImg, pooling)
        style_pre = np.array([vgg.preprocess(style, vgg_mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={styleImg: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        noise = np.random.normal(size=content_shape, scale=np.std(content) * 0.1)
        initial = tf.random_normal(content_shape) * 0.256
        inferenceImg = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, inferenceImg, pooling)

        # compute content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                net[content_layer] - content_features[content_layer]) / content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)

        # compute style loss
        style_loss = 0
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(
                style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss += style_weight * style_blend_weight * reduce(tf.add, style_losses)

        # skip compute variation denoise, in order to shorten the running time
        # total variation denoising
        # tv_y_size = _tensor_size(inferenceImg[:, 1:, :, :])
        # tv_x_size = _tensor_size(inferenceImg[:, :, 1:, :])
        # tv_loss = tv_weight * 2 * (
        #         (tf.nn.l2_loss(inferenceImg[:, 1:, :, :] - inferenceImg[:, :content_shape[1] - 1, :, :]) /
        #          tv_y_size) +
        #         (tf.nn.l2_loss(inferenceImg[:, :, 1:, :] - inferenceImg[:, :, :content_shape[2] - 1, :]) /
        #          tv_x_size))

        tv_loss = 0
        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer training
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            for i in range(iterations):
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = inferenceImg.eval()

                    img_out = vgg.unprocess(best.reshape(content_shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(
                            Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))

                    yield ((None if last_step else i), img_out)


#############################
#   MAIN
#############################
def main():
    # Output File
    checkpoint_imgName = "out/home_cat_%s.jpg"  # CheckPoint image name
    inference_imgName = "out/home_cat.jpg"  # Final output inference image name

    # Input Params
    VGG19 = "imagenet-vgg-verydeep-19.mat" # use VGGNet19 model to training.
    style_img = imgRead("stylePics/home.jpg")
    content_img = imgRead("contentPics/cat.jpg")

    style_img = scipy.misc.imresize(style_img, STYLE_SCALE * content_img.shape[1] / style_img.shape[1])
    initial_img = content_img

    stime = time.clock()
    for iteration, combinedImg in inferenceImg(network=VGG19, initial_img=initial_img,
                                               initial_noiseblend=INITIAL_NOISEBLEND,
                                               content=content_img, style=style_img, preserve_colors=PRESERVE_COLORS,
                                               iterations=ITERATIONS,
                                               content_weight=CONTENT_WEIGHT, content_weight_blend=CONTENT_WEIGHT_BLEND,
                                               style_weight=STYLE_WEIGHT, style_layer_weight_exp=STYLE_LAYER_WEIGHT_EXP,
                                               style_blend_weight=STYLE_BLEND_WEIGHT,
                                               tv_weight=TV_WEIGHT, learning_rate=LEARNING_RATE, beta1=BETA1,
                                               beta2=BETA2, epsilon=EPSILON, pooling=POOLING,
                                               print_iterations=PRINT_ITERATIONS,
                                               checkpoint_iterations=CHECKPOINT_ITERATIONS
                                               ):
        imgFileName = None
        if iteration is not None:
            imgFileName = checkpoint_imgName % str(iteration)
        else:
            imgFileName = inference_imgName

        if imgFileName:
            imgSave(imgFileName, combinedImg)
    etime = time.clock()
    stderr.write("total cost time: " + (etime-stime)/10000)


if __name__ == "__main__":
    main()
