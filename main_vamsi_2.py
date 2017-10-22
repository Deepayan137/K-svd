# from time import time
import cv2
import argparse
# import matplotlib.pyplot as plt
import numpy as np
from operator import mul, sub
from skimage.util.shape import *
from skimage.util import pad
from functools import reduce
from math import floor, sqrt
from scipy.stats import chi2
import timeit
import sys
import pdb
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
# import scipy as sp
# import pdb
# from sklearn.feature_extraction.image import extract_patches_2d
# from sklearn.feature_extraction.image import reconstruct_from_patches_2d
# from sklearn.datasets import make_sparse_coded_signal
# from sklearn.decomposition import MiniBatchDictionaryLearning
# from sklearn.linear_model import OrthogonalMatchingPursuit
# from matplotlib import pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

patch_size = 8


sigma = 10                 # Noise standard dev.

window_shape = (patch_size, patch_size)    # Patches' shape
step = 8                  # Patches' step
ratio = 0.1            # Ratio for the dictionary (training set).
ksvd_iter = 5   


#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- PATCH CREATION -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def patch_matrix_windows(img, window_shape, step):
    # we return an array of patches(patch_size X num_patches)
    patches = view_as_windows(img, window_shape, step=step)  # shape = [patches in image row,patches in image col,rows in patch,cols in patch]
    # size of cond_patches = patch size X number of patches
    cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            cond_patches[:, j+patches.shape[1]*i] = np.concatenate(patches[i, j], axis=0)
    return cond_patches, patches.shape


def image_reconstruction_windows(mat_shape, patch_mat, patch_sizes, step):
    img_out = np.zeros(mat_shape)
    for l in range(patch_mat.shape[1]):
        i, j = divmod(l, patch_sizes[1])
        temp_patch = patch_mat[:, l].reshape((patch_sizes[2], patch_sizes[3]))
        img_out[i*step:(i+1)*step, j*step:(j+1)*step] = temp_patch[:step, :step].astype(int)
    return img_out

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ APPROXIMATION PURSUIT METHOD : -----------------------------------------#
#------------------------------------- MULTI-CHANNEL ORTHOGONAL MATCHING PURSUIT -----------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def single_channel_omp(phi, vect_y, sigma):

    vect_sparse = np.zeros(phi.shape[1])
    res = np.linalg.norm(vect_y)            # energy in the signal----sqrt( sum( square of signal elements ) )
    atoms_list = []

    # print(phi.shape,vect_sparse.shape)
    while res/sigma > sqrt(chi2.ppf(0.995, vect_y.shape[0] - 1)) and len(atoms_list) < phi.shape[1]:
        vect_c = phi.T.dot(vect_y - phi.dot(vect_sparse))
        i_0 = np.argmax(np.abs(vect_c))
        atoms_list.append(i_0)
        vect_sparse[i_0] += vect_c[i_0]

        # Orthogonal projection.
        index = np.where(vect_sparse)[0]
        vect_sparse[index] = np.linalg.pinv(phi[:, index]).dot(vect_y)
        res = np.linalg.norm(vect_y - phi.dot(vect_sparse))

    return vect_sparse, atoms_list



#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- DICTIONARY UPDATING METHODS -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def dict_update(phi, matrix_y, matrix_sparse, k):
    indexes = np.where(matrix_sparse[k, :] != 0)[0]
    phi_temp = phi
    sparse_temp = matrix_sparse

    if len(indexes) > 0:
        phi_temp[:, k][:] = 0

        matrix_e_k = matrix_y[:, indexes] - phi_temp.dot(sparse_temp[:, indexes])
        u, s, v = svds(np.atleast_2d(matrix_e_k), 1)

        phi_temp[:, k] = u[:, 0]
        sparse_temp[k, indexes] = np.asarray(v)[0] * s[0]
    return phi_temp, sparse_temp


def approx_update(phi, matrix_y, matrix_sparse, k):
    indexes = np.where(matrix_sparse[k, :] != 0)[0]
    phi_temp = phi

    if len(indexes) > 0:
        phi_temp[:, k] = 0
        vect_g = matrix_sparse[k, indexes].T
        vect_d = (matrix_y - phi_temp.dot(matrix_sparse))[:, indexes].dot(vect_g)
        vect_d /= np.linalg.norm(vect_d)
        vect_g = (matrix_y - phi_temp.dot(matrix_sparse))[:, indexes].T.dot(vect_d)
        phi_temp[:, k] = vect_d
        matrix_sparse[k, indexes] = vect_g.T
    return phi_temp, matrix_sparse


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def k_svd(phi, matrix_y, sigma, algorithm, n_iter, approx='yes'):
    phi_temp = phi
    #pdb.set_trace()                                              # X(size mxp) = phi(size mxn) x maatrix_sparse(size nxp)
    matrix_sparse = np.zeros((phi.T.dot(matrix_y)).shape)       # initializing spare matrix
    n_total = []

    print ('\nK-SVD, with residual criterion.')
    print ('-------------------------------')

    for k in range(n_iter):
        print ("Stage " , str(k+1) , "/" , str(n_iter) , "...")

        def sparse_coding(f):
            t = f+1
            print("Stage " , str(k+1),"- Sparse coding : Channel", t)
            return algorithm(phi_temp, matrix_y[:, f], sigma)[0]

        sparse_rep = list(map(sparse_coding, range(matrix_y.shape[1])))
        matrix_sparse = np.array(sparse_rep).T
        print(matrix_sparse.shape)
        count = 1

        updating_range = phi.shape[1]

        for j in range(updating_range):
            r = floor(count/float(updating_range)*100)
            print("Stage " , str(k+1),"- Dictionary updating :",r, "%")
            if approx == 'yes':
                phi_temp, matrix_sparse = approx_update(phi_temp, matrix_y, matrix_sparse, j)
            else:
                phi_temp, matrix_sparse = dict_update(phi_temp, matrix_y, matrix_sparse, j)
            count += 1
        print ('\r- Dictionary updating complete.\n')

    return phi_temp, matrix_sparse, n_total


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ DENOISING METHOD -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def denoising(noisy_image, learning_image, window_shape, window_step, sigma, learning_ratio=0.1, ksvd_iter=1):

    # 1. Form noisy patches.
    padded_noisy_image = pad(noisy_image, pad_width=window_shape, mode='symmetric')
    noisy_patches, noisy_patches_shape = patch_matrix_windows(padded_noisy_image, window_shape, window_step)
    padded_lea_image = pad(learning_image, pad_width=window_shape, mode='symmetric')
    lea_patches, lea_patches_shape = patch_matrix_windows(padded_lea_image, window_shape, window_step)
    # noisy_patches = extract_patches_2d(noisy_image, window_shape)
    # noisy_patches = noisy_patches.reshape(noisy_patches.shape[0], -1).T
    print ('Shape of dataset    : ' , str(noisy_patches.shape))

    # 2. Form explanatory dictionary.
    k = int(learning_ratio*lea_patches.shape[1])        # number of dictionary elements
    # k = 256
    indexes = np.random.random_integers(0, noisy_patches.shape[1]-1, k)   # indexes of patches for dictionary elements

    # dictionary intialization
    basis = lea_patches[:, indexes]  
    # basis = noisy_patches[:, indexes]            # each column is a new atom
    basis /= np.sum(basis.T.dot(basis), axis=-1)    # dictionary normalization

    print( 'Shape of dictionary : ' , str(basis.shape) + '\n')

    # 3. Compute K-SVD.
    start = timeit.default_timer()
    basis_final, sparse_final, n_total = k_svd(basis, noisy_patches, sigma, single_channel_omp, ksvd_iter)
    stop = timeit.default_timer()
    print ("Calculation time : " , str(stop - start) , ' seconds.')

    # 4. Reconstruct the image.
    patches_approx = basis_final.dot(sparse_final)
    padded_denoised_image = image_reconstruction_windows(padded_noisy_image.shape,
                                                         patches_approx, noisy_patches_shape, window_step)  
    # patches_approx = patches_approx.reshape(patches_approx.shape[1], *(8,8))                                                  
    # denoised_image = reconstruct_from_patches_2d(patches_approx, (noisy_image.shape[0], noisy_image.shape[1]))

    shrunk_0, shrunk_1 = tuple(map(sub, padded_denoised_image.shape, window_shape))
    denoised_image = np.abs(padded_denoised_image)[window_shape[0]:shrunk_0, window_shape[1]:shrunk_1]
    return denoised_image, stop - start, n_total






image = cv2.imread(args['image'], 0)
image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
print(image.shape)
# image = image / 255.
# image = image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]
# image /= 4.0
# noisy_image = image.copy()
# noisy_image += 0.075 * np.random.randn(image.shape[0], image.shape[1])
noise_layer = np.random.normal(0, sigma ^ 2, image.size).reshape(image.shape).astype(int)
noisy_image = image + noise_layer
learning_image = noisy_image.copy()
# learning_image = image.copy()

# cv2.imshow('orignal', image)
# cv2.imshow('noise_layer', noise_layer.astype('uint8'))
# cv2.imshow('noisy_image', noisy_image)
# cv2.imshow('learning_image', learning_image.astype('uint8'))

denoised_image, calc_time, n_total = denoising(noisy_image, learning_image, window_shape, step, sigma, ratio, ksvd_iter)

# denoising(noisy_image, learning_image, window_shape, step, sigma, ratio, ksvd_iter)
#pdb.set_trace()

cv2.imshow('orignal', image)
cv2.imshow('noisy_image', noisy_image.astype('uint8'))
cv2.imshow('denoised_image', denoised_image.astype('uint8'))




cv2.waitKey(0)
# while cv2.waitKey(-1) == 27:
#     print('break')
#     break
