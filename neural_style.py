import numpy as np
import scipy.misc

from PIL import Image

import tensorflow as tf

#from network import upsample_kwh
from network import residual_padding

from stylize import stylize

# default arguments

LEARNING_RATE2 = 1e-3
EPOCHS = 400

CONTENT_IN = 'E:/paper/image_data/%07d.jpg'
RESULT_IN = 'E:/paper/the_scream/%07d.jpg'

CHECK_POINT_PATH = 'E:/paper/test/the_scream/%07d_%03d.jpg'
TEST_IN = './content/stata.jpg'
OUT_PATH = './the_scream_stata.jpg'
SAVE_PATH = "./saver/the_scream.ckpt"

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 50
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

#CONTENT_IN = './content.jpg'
STYLE_IN = ['./style/the_scream.jpg']
OUTPUT = './result/the_scream_stata_final.jpg'
CHECK_POINT = 10
FINAL_CHECK_POINT_PATH = './result/the_scream_stata/%s.jpg'
PRINT_ITERATIONS = None #100
PRESERVE_COLORS = False

STYLE_BLEND_WEIGHTS = None


def main():

    content_size = imread(CONTENT_IN % 0)

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=(1,) + content_size.shape, name="X_content")
        Y_result = tf.placeholder(tf.float32, shape=(1,) + content_size.shape, name="Y_result")

        COST_AVG = tf.placeholder(tf.float32, name="COST_AVG")
        cost_avg_summ = tf.summary.scalar("cost_avg", COST_AVG)
        preds = residual_padding.net(X_content / 255.0)

        cost = tf.reduce_mean(tf.square(tf.subtract(preds[0], Y_result)))
        #cost_summ = tf.summary.scalar("cost", cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE2).minimize(cost)

        writer = tf.summary.FileWriter("./logs/the_scream", sess.graph)
        merged = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        try:
            # load parameter
            saver.restore(sess, SAVE_PATH)
            print("load")
        except:
            print("first training")

        pre_cost_avg = 10000
        best_cost = 10000
        for epoch in range(EPOCHS):
            cost_avg = 0
            for num in range(3000):

                content_image = imread(CONTENT_IN % num)
                result_image = imread(RESULT_IN % (num))

                feed_dict = {
                    X_content: [content_image],
                    Y_result: [result_image]
                }

                _, COST, test_image = sess.run([optimizer, cost, preds], feed_dict=feed_dict)
                print("epoch : %03d, num : %04d, cost : %f" % (epoch ,num, COST))
                cost_avg += COST
                if(num % 100 == 0):
                    imsave(CHECK_POINT_PATH % (num, epoch),test_image[0] )
            cost_avg /= 3000
            if (best_cost > cost_avg):
                best_cost = cost_avg
            print ("error_mean : %f, error_dif : %f, best_cost : %f" %(cost_avg, cost_avg - pre_cost_avg, best_cost))
            pre_cost_avg = cost_avg
            saver.save(sess, SAVE_PATH)

            feed_dict3 = {
                COST_AVG: cost_avg
            }

            summary = sess.run(merged, feed_dict = feed_dict3)
            writer.add_summary(summary, epoch)


def test(content_image, result_path, save_path):
    CONTENT_IMAGE = imread(content_image)
    image_shape = (1,) + CONTENT_IMAGE.shape
    input_image = [CONTENT_IMAGE]
    style_images = [imread(style) for style in STYLE_IN]

    target_shape = CONTENT_IMAGE.shape

    for i in range(len(style_images)):
        style_scale = STYLE_SCALE

        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                                              target_shape[1] / style_images[i].shape[1])
    style_blend_weights = STYLE_BLEND_WEIGHTS
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0 / len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight
                               for weight in style_blend_weights]

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=image_shape, name="X_content")

        preds = residual_padding.net(X_content / 255.0)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path)

        feed_dict = {
            X_content: input_image
        }

        result = preds.eval(feed_dict = feed_dict)

        initial = result[0]
        initial_noiseblend = 0.0
        for iteration, image in stylize(
                network=VGG_PATH,
                initial=initial,
                initial_noiseblend=initial_noiseblend,
                content=CONTENT_IMAGE,
                styles=style_images,
                preserve_colors=PRESERVE_COLORS,
                iterations=ITERATIONS,
                content_weight=CONTENT_WEIGHT,
                content_weight_blend=CONTENT_WEIGHT_BLEND,
                style_weight=STYLE_WEIGHT,
                style_layer_weight_exp=STYLE_LAYER_WEIGHT_EXP,
                style_blend_weights=style_blend_weights,
                tv_weight=TV_WEIGHT,
                learning_rate=LEARNING_RATE,
                beta1=BETA1,
                beta2=BETA2,
                epsilon=EPSILON,
                pooling=POOLING,
                print_iterations=PRINT_ITERATIONS,
                checkpoint_iterations=CHECK_POINT
        ):
            output_file = None
            combined_rgb = image
            if iteration is not None:
                if CHECK_POINT_PATH:
                    output_file = FINAL_CHECK_POINT_PATH % iteration
            else:
                output_file = OUTPUT
            if output_file:
                imsave(output_file, combined_rgb)

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    #main()
    test(TEST_IN, OUT_PATH ,SAVE_PATH)