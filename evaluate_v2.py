"""Learned primal-dual method."""
import os
import h5py
import time
from os.path import join, exists
from skimage import io
import tensorflow as tf
import numpy as np
import scipy.io as sio
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

from train_v2 import generate_data
from dataset import get_test_data
from model import getMultiCoilImage, getCoilCombineImage

def get_data_sos(label, mask_t, bacth_size_mask=4):
    batch, nx, ny, nt, coil = label.shape
    nx, ny, nt = mask_t.shape
    mask_t = np.transpose(mask_t, (2, 0, 1))
    #mask = mask_t[0:bacth_size_mask, ...]
    mask = np.tile(mask_t[:, :, :, np.newaxis], (1, 1, 1, coil)) #nt, nx, ny, coil

    label = np.squeeze(label)
    label = np.transpose(label, (2, 3, 0, 1)) # nt, coil, nx, ny
    k_full_shift = fft2(label, axes=(-2, -1)) # batch, coil, nx, ny
    #k_full_shift = np.tile(k_full_shift, (bacth_size_mask, 1, 1, 1))
    k_full_shift = np.transpose(k_full_shift, (0, 2, 3, 1)) # nt, nx, ny, coil
    k_und_shift = k_full_shift * mask
    label_sos = np.sum(abs(label**2), axis=1)**(1/2)
    #label_sos = np.tile(label_sos, [bacth_size_mask, 1, 1])
    mask = mask[:, :, :, 0]
    return k_und_shift, label_sos, mask


def evaluate(test_data, mask_t, model_save_path, model_file):
    result_dir = os.path.join('results', model_file+'_test_on_uniform_mask' )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with tf.Graph() .as_default() as g:
        y_m = tf.placeholder(tf.complex64, (None, 192, 192, 20), "y_m")
        mask = tf.placeholder(tf.complex64, (None, 192, 192), "mask")
        x_true = tf.placeholder(tf.float32, (None, 192, 192), "x_true")

        x_pred = getCoilCombineImage(y_m, mask, n_iter=8)

        residual = x_pred - x_true
        #residual = tf.stack([tf.real(residual), tf.imag(residual)], axis=4)
        loss = tf.reduce_mean(residual ** 2)

        with tf.Session() as sess:

            #ckpt = tf.train.get_checkpoint_state(model_save_path)
            saver = tf.train.Saver()

            #if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_save_path)

            count = 0
            recon_total = np.zeros((test_data.shape[0], test_data.shape[3], test_data.shape[1], test_data.shape[2]))

            for ys in generate_data(test_data, BATCH_SIZE=1, shuffle=False):
                und_kspace, label, mask_d = get_data_sos(ys, mask_t)
                im_start = time.time()
                loss_value, pred = sess.run([loss, x_pred],
                                                     feed_dict={y_m: und_kspace,
                                                                mask: mask_d,
                                                                x_true: label})
                recon_total[count, ...] = pred
                count += 1
                sio.savemat(join(result_dir, 'recon_%d.mat' % count), {'im_recon': pred})

                print("The loss of No.{} test data = {}".format(count, loss_value))
            #sio.savemat(join(result_dir, 'recon_total.mat'), {'recon': recon_total})


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_dir = '/data0/ziwen/data/h5/multi_coil_strategy_v0'
    test_data = get_test_data(data_dir)
    #mask_t = sio.loadmat(join('mask', 'UIH_TIS_mask_t_192_192_X400_ACS_16_R_3.56.mat'))['mask_t']
    #mask_t = sio.loadmat(join('mask', 'random_gauss_mask_t_192_192_16_ACS_16_R_3.64.mat'))['mask_t']
    #mask_t = sio.loadmat(join('mask', 'random_gauss_mask_t_192_192_16_ACS_16_R_5.42.mat'))['mask_t']
    mask_t = sio.loadmat(join('mask', 'UIH_TIS_mask_t_192_192_X400_ACS_16_R_3.56.mat'))['mask_t']
    mask_t = np.fft.fftshift(mask_t, axes=(0, 1))
    acc = mask_t.size / np.sum(mask_t)
    print('Acceleration Rate:{:.2f}'.format(acc))

    project_root = '.'
    model_file = "Unsupervised learning via TIS_mask_5t_multi-coil_2149_train_on_random_mask_AMAX"
    model_name = "Unsupervised learning via TIS_mask_5t_multi-coil_2149_train_on_random_mask_AMAX.ckpt"
    model = join(project_root, 'checkpoints/%s' % model_file, model_name)
    evaluate(test_data, mask_t, model_save_path=model, model_file=model_file)


if __name__ == '__main__':
    tf.app.run()