#coding:utf-8
import tensorflow as tf
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.misc as scm
'''
Function introduction:
dir:          the data direction, for example:'/home/liuvv/Desktop/picture'
batch_size:   the batch size
scle:         defult is False, if True, the result data is crop from raw data
scale_size:   the data size of result data of we need
is_gratscale: defult is False, if True, the result data is gratscale pic
'''
def get_loader(dir, batch_size, scale_size, scale=False,is_gratscale=False):

    dataser_dir = os.path.basename(dir)
    for ext in ["jpg", "png"]:
        path = glob("{}/*.{}".format(dir, ext))
        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

    with Image.open(path[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_quene = tf.train.string_input_producer(list(path), shuffle=False, seed=None)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_quene)
    image = tf_decode(data, channels=3)
    if is_gratscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)
    quene = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=5000+3*batch_size,
                                   min_after_dequeue=5000, name='synthetic_inputs')
    if scale:
        quene = tf.image.crop_to_bounding_box(quene, 0, 0, 64, 64)
        quene = tf.image.resize_nearest_neighbor(quene, [scale_size, scale_size])
    else :
        quene = tf.image.resize_nearest_neighbor(quene, [scale_size, scale_size])

    image = tf.to_float(quene)
    image = image/255. - 0.5

    return image


'''
Function introduction:
dir:        the data direction, for example:'/home/liuvv/Desktop/picture'
batch_size: the batch size
idx:        the index from [0, iters]
'''
def load_data(data_dir, batch_size, idx):
    def get_image(img_path):
        img = scm.imread(img_path) / 255. - 0.5

        # img = img[..., ::-1]  # rgb to bgr
        return img

    data = sorted(glob(os.path.join(data_dir, "*.*")))
    # random_order = np.random.permutation(len(data))
    # data = [data[i] for i in random_order[:]]
    batch_files = data[idx * batch_size: (idx + 1) * batch_size]
    batch_data = [get_image(batch_file) for batch_file in batch_files]
    return batch_data

if __name__ == '__main__':

    img = load_data('/home/liuvv/Desktop/0', 2, 0)
    print 'data loading done'
    print img
    # with tf.Session() as sess:
        # img = sess.run(img)
    img = (img+np.float32(0.5))*255
    plt.imsave('dog2.png', img[0])
    plt.imsave('cat2.png', img[1])


'''#test 
    img = get_loader('/home/liuvv/Desktop/0', 2, 64)
    print 'data loading done'
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # with tf.Session() as sess:
    img = (img+0.5)*255.0
    img = sess.run(img)
    print img

    plt.imsave('dog1.png', img[0])
    # plt.figure(2)
    # plt.imshow(img[1])
    # plt.show()
    plt.imsave('cat1.png', img[1])
    coord.request_stop()
    coord.join(threads)
    '''