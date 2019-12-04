import h5py
import numpy as np
import os


def get_train_data(data_dir):
    with h5py.File(os.path.join(data_dir, './train_real.h5')) as f:
        label_real = f['train_real'][:]
    num, coil, ny, nx = label_real.shape
    #data_real = np.transpose(data_real, (0, 2, 1))
    with h5py.File(os.path.join(data_dir, './train_imag.h5')) as f:
        label_imag = f['train_imag'][:]
    #data_imag = np.transpose(kspace_imag, (0, 2, 1))
    label = label_real + 1j * label_imag
    label = np.transpose(label, (0, 3, 2, 1))

    num_train = 1900
    num_validate = 249
    train_label = label[0:num_train]
    validate_label = label[num_train:num_train + num_validate]
    return train_label, validate_label

def get_test_data(data_dir):
    with h5py.File(os.path.join(data_dir, './test_real_45.h5')) as f:
        test_real = f['test_real'][:]
    num, nc, nt, ny, nx = test_real.shape
    #data_real = np.transpose(data_real, (0, 2, 1))
    with h5py.File(os.path.join(data_dir, './test_imag_45.h5')) as f:
        test_imag = f['test_imag'][:]
    #data_imag = np.transpose(kspace_imag, (0, 2, 1))
    test_label = test_real + 1j * test_imag
    test_label = np.transpose(test_label, (0, 4, 3, 2, 1))

    return test_label

def get_fine_tuning_data(data_dir):
    with h5py.File(os.path.join(data_dir, './fine_tuning_real.h5')) as f:
        fine_tuning_real = f['fine_tuning_real'][:]
    num, nc, nt, ny, nx = fine_tuning_real.shape
    #data_real = np.transpose(data_real, (0, 2, 1))
    with h5py.File(os.path.join(data_dir, './fine_tuning_imag.h5')) as f:
        fine_tuning_imag = f['fine_tuning_imag'][:]
    #data_imag = np.transpose(kspace_imag, (0, 2, 1))
    fine_tuning_label = fine_tuning_real + 1j * fine_tuning_imag
    fine_tuning_label = np.transpose(fine_tuning_label, (0, 4, 3, 2, 1))

    return fine_tuning_label

def get_train_data_UIH(data_dir):
    with h5py.File(os.path.join(data_dir, './UIH_Data.h5')) as f:
        UIH_real = f['trnData_real'][:]
        UIH_img = f['trnData_img'][:]
        mask_t = f['Mask_1D_x4'][:]
    UIH_data = UIH_real + 1j * UIH_img
    UIH_data = np.transpose(UIH_data, (0, 2, 3, 1))

    with h5py.File(os.path.join(data_dir, './trnData_CUBE.hdf5')) as f:
        GE_real = f['trnData_real'][:]
        GE_img = f['trnData_img'][:]
    GE_data = GE_real + 1j * GE_img
    GE_data = np.transpose(GE_data, (0, 2, 3, 1))

    kspace = np.concatenate((UIH_data, GE_data))

    # num_train = 500
    # num_validate = 110
    # train_kspace = kspace[0:num_train]
    # validate_kspace = kspace[num_train:num_train + num_validate]
    return kspace, mask_t