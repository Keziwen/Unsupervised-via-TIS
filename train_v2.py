import tensorflow as tf
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift

import os
from skimage import io
import time

from dataset import get_train_data
from model import getMultiCoilImage, getCoilCombineImage, getCoilCombineImage_DCCNN
import scipy.io as sio
from os.path import join

def generate_data(x, BATCH_SIZE=1, shuffle=False):
    """Generate a set of random data."""
    n = len(x)
    if shuffle:
        x = np.random.permutation(x)

    for j in range(0, n, BATCH_SIZE):
        yield x[j:j+BATCH_SIZE]


def get_data_sos(label, mask_t, bacth_size_mask=4):
    batch, nx, ny, coil = label.shape
    nx, ny, nt = mask_t.shape
    mask_t = np.transpose(mask_t, (2, 0, 1))
    mask = mask_t[0:bacth_size_mask, ...]
    mask = np.tile(mask[:, :, :, np.newaxis], (1, 1, 1, coil)) #batch_size_mask, nx, ny, coil

    label = np.transpose(label, (0, 3, 1, 2))
    k_full_shift = fft2(label, axes=(-2, -1)) # batch, coil, nx, ny
    k_full_shift = np.tile(k_full_shift, (bacth_size_mask, 1, 1, 1))
    k_full_shift = np.transpose(k_full_shift, (0, 2, 3, 1)) # batch_size_mask, nx, ny, coil
    k_und_shift = k_full_shift * mask
    label_sos = np.sum(abs(label**2), axis=1)**(1/2)
    label_sos = np.tile(label_sos, [bacth_size_mask, 1, 1])
    mask = mask[:, :, :, 0]
    return k_und_shift, label_sos, mask

if __name__ == "__main__":
    lr_base = 1e-03
    BATCH_SIZE = 1
    lr_decay_rate = 0.98
    # EPOCHS = 200
    num_epoch = 200
    num_train = 1900
    num_validate = 249
    Nx = 192
    Ny = 192
    Nc = 20

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    base_dir = '.'
    name = 'Unsupervised learning via TIS_mask_5t_multi-coil_2149_train_on_random_4X_DC_CNN_AMAX'
    # name = os.path.splitext(os.path.basename(__file__))[0]
    # model_save_path = os.path.join(base_dir, name)
    # if not os.path.isdir(model_save_path):
    #     os.makedirs(model_save_path)

    checkpoint_dir = os.path.join(base_dir, 'checkpoints/%s' % name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, '{}.ckpt'.format(name))

    # data for train
    data_dir = '/data0/ziwen/data/h5/multi_coil_strategy_v0'
    train_label, validate_label = get_train_data(data_dir)
    #train_label = np.zeros([100, 192, 192, 20])
    #validate_label = np.zeros([100, 192, 192, 20])

    mask_t = sio.loadmat(join('mask', 'random_gauss_mask_t_192_192_16_ACS_16_R_3.64.mat'))['mask_t']
    mask_t = np.fft.fftshift(mask_t, axes=(0, 1))
    print('Acceleration factor: {}'.format(mask_t.size/float(mask_t.sum())))
    # mk = sio.loadmat('Random1D_256_256_R6.mat')
    # mask_t = np.fft.fftshift(mk['mask'], axes=(-1, -2))

    y_m = tf.compat.v1.placeholder(tf.complex64, (None, Nx, Ny, Nc), "y_m")
    mask = tf.compat.v1.placeholder(tf.complex64, (None, Nx, Ny), "mask")
    x_true = tf.compat.v1.placeholder(tf.float32, (None, Nx, Ny), "x_true")

    x_pred = getCoilCombineImage_DCCNN(y_m, mask, n_iter=8)

    with tf.name_scope("loss"):
        residual = x_pred - x_true
        #residual = tf.stack([tf.real(residual_cplx), tf.imag(residual_cplx)], axis=4)
        Y = tf.reduce_mean(residual ** 2)
        loss = Y

    global_step = tf.Variable(0., trainable=False)
    lr = tf.compat.v1.train.exponential_decay(lr_base,
                                    global_step=global_step,
                                    decay_steps=num_train // BATCH_SIZE,
                                    decay_rate=lr_decay_rate,
                                    staircase=False)
    with tf.name_scope("train"):
        train_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    saver = tf.compat.v1.train.Saver()
    with tf.Session() as sess:


        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # saver = tf.train.Saver()
        # if ckpt and ckpt.model_checkpoint_path:
        #saver.restore(sess, checkpoint_path)
        train_plot = []
        validate_plot = []

        # train the network
        for i in range(num_epoch):
            count_train = 0
            loss_sum_train = 0.0
            for ys in generate_data(train_label, BATCH_SIZE=BATCH_SIZE, shuffle=True):
                train, label, mask_d = get_data_sos(ys, mask_t)
                im_start = time.time()
                _, loss_value, step, pred = sess.run([train_step, loss, global_step, x_pred],
                                                     feed_dict={y_m: train,
                                                                mask: mask_d,
                                                                x_true: label})
                im_end = time.time()
                loss_sum_train += loss_value
                print("{}\{}\{} of training loss:\t\t{:.6f} \t using :{:.4f}s".
                      format(i + 1, count_train + 1, int(num_train / BATCH_SIZE),
                             loss_sum_train / (count_train + 1), im_end - im_start))
                count_train += 1

            count_validate = 0
            loss_sum_validate = 0.0
            for ys_validate in generate_data(validate_label, shuffle=True):
                y_rt_validate, x_true_validate, mask_validate = get_data_sos(ys_validate, mask_t)
                im_start = time.time()
                loss_value_validate = sess.run(loss, feed_dict={y_m: y_rt_validate,
                                                                mask: mask_validate,
                                                                x_true: x_true_validate})
                im_end = time.time()
                loss_sum_validate += loss_value_validate
                count_validate += 1
                print("{}\{}\{} of validation loss:\t\t{:.6f} \t using :{:.4f}s".
                      format(i + 1, count_validate, int(num_validate / BATCH_SIZE),
                             loss_sum_validate / count_validate, im_end - im_start))
            # train_plot.append(loss_sum_train / count_train_per)
            # validate_plot.append(loss_sum_validate / count_validate)
            saver.save(sess, checkpoint_path)
    # train_plot_name = 'train_plot.npy'
    # np.save(os.path.join(checkpoint_dir, train_plot_name), train_plot)
    # validate_plot_name = 'validate_plot.npy'
    # np.save(os.path.join(checkpoint_dir, validate_plot_name), validate_plot)
