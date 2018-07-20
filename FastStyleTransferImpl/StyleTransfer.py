import numpy as np
import imageio
from PIL import Image
from transferNet import net
import tensorflow as tf

###########################################
# Gobal Const and fine tuning params define
###########################################
VGG_MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])


###########################
#   Tool func
###########################
def imgRead(path):
    img = imageio.imread(path).astype(np.float)
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))

    return img


def imgSave(path, image):
    img = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


#########################################
#   inference image
#########################################
def inferenceImg(image, outputImg, checkpoint_dir):
    img_pre = np.array([image - VGG_MEAN_PIXEL])

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session(config=soft_config) as sess:
        batch_shape = (1,) + image.shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name="img_placeholder")
        preds = net(img_placeholder)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_dir)
        _preds = sess.run(preds, feed_dict = {img_placeholder: img_pre})
        imgSave(outputImg, _preds[0])


#############################
#   MAIN
#############################
def main():
    inputImg = "inputPics/dog.jpg"      # input image.
    outpath = "outputPics/inf_dog.jpg"  # output dir
    checkpoint_dir = "checkpoint/rain-princess.ckpt"

    image = imgRead(inputImg)
    inferenceImg(image, outpath, checkpoint_dir)


if __name__ == "__main__":
    main()