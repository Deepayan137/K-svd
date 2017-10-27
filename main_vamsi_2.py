# from time import time
import cv2
import argparse
# import matplotlib.pyplot as plt
import numpy as np
from operator import mul, sub
from skimage.util.shape import *
from skimage.util import pad
from functools import reduce
from math import floor, sqrt, log10
from scipy.stats import chi2
from scipy.sparse.linalg import svds
import timeit
import sys
import pdb
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
# import scipy as sp
# import pdb
# from sklearn.feature_extraction.image import extract_patches_2d
# from sklearn.datasets import make_sparse_coded_signal
# from sklearn.decomposition import MiniBatchDictionaryLearning
# from sklearn.linear_model import OrthogonalMatchingPursuit
# from matplotlib import pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

patch_size = 8

sigma = 20                 # Noise standard dev.

window_shape = (patch_size, patch_size)    # Patches' shape
window_stride = 4                  # Patches' step
dict_ratio = 0.1            # Ratio for the dictionary (training set).
num_dict=256   
ksvd_iter = 20   

max_resize_dim = 256
dict_train_blocks = 65000



#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- PATCH CREATION -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def patch_matrix_windows(img, stride):
    # we return an array of patches(patch_size X num_patches)
    patches = view_as_windows(img, window_shape, step=stride)  # shape = [patches in image row,patches in image col,rows in patch,cols in patch]
    # size of cond_patches = patch size X number of patches
    cond_patches = np.zeros((reduce(mul, patches.shape[2:4]), reduce(mul, patches.shape[0:2])))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            cond_patches[:, j+patches.shape[1]*i] = np.concatenate(patches[i, j], axis=0)
    return cond_patches, patches.shape


def reconstruct_image(input_shape, patch_final, noisy_image):
    img_out = np.zeros(input_shape)
    weight = np.zeros(input_shape)
    num_blocks = input_shape[0] - patch_size + 1
    for l in range(patch_final.shape[1]):
        i, j = divmod(l, num_blocks)
        temp_patch = patch_final[:, l].reshape(window_shape)
        # img_out[i, j] = temp_patch[1, 1]
        img_out[i:(i+patch_size), j:(j+patch_size)] = img_out[i:(i+patch_size), j:(j+patch_size)] + temp_patch
        weight[i:(i+patch_size), j:(j+patch_size)] = weight[i:(i+patch_size), j:(j+patch_size)] + np.ones(window_shape)      

    # img_out = img_out/weight
    img_out = (noisy_image+0.034*sigma*img_out)/(1+0.034*sigma*weight);


    print('max: ',np.max(img_out))

    return img_out


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ APPROXIMATION PURSUIT METHOD : -----------------------------------------#
#------------------------------------- MULTI-CHANNEL ORTHOGONAL MATCHING PURSUIT -----------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def omp(D, data):
    max_error = sqrt(((sigma**1.15)**2)*D.shape[0])
    max_coeff = D.shape[0]/2

    sparse_coeff = np.zeros((D.shape[1],data.shape[1]))
    tot_res = 0
    for i in range(data.shape[1]):
        count = floor((i+1)/float(data.shape[1])*100)
        sys.stdout.write("\r- Sparse coding : Channel : %d%%" % count)
        sys.stdout.flush()
        
        x = data[:,i]
        res = x
        atoms_list = []
        indx = []
        res_norm = np.linalg.norm(res)
        temp_sparse = np.zeros(D.shape[1])

        while res_norm > max_error and len(atoms_list) < max_coeff:
            proj = D.T.dot(res)
            i_0 = np.argmax(np.abs(proj))
            atoms_list.append(i_0)

            temp_sparse = np.linalg.pinv(D[:,atoms_list]).dot(x)
            res = x - D[:,atoms_list].dot(temp_sparse)
            res_norm = np.linalg.norm(res)

        tot_res += res_norm
        if len(atoms_list) > 0:
            sparse_coeff[atoms_list, i] = temp_sparse
    print('\n',tot_res)
    print ('\r- Sparse coding complete.\n')

    return sparse_coeff

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- DICTIONARY METHODS -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def dict_initiate(noisy_image, stride):
    # dictionary intialization
    noisy_patches, noisy_patches_shape = patch_matrix_windows(noisy_image, stride)

    indexes = np.random.random_integers(0, noisy_patches.shape[1]-1, num_dict)   # indexes of patches for dictionary elements
    dict_init = np.array(noisy_patches[:, indexes])            # each column is a new atom

    # dictionary normalization
    dict_init = dict_init - dict_init.mean()
    temp = np.diag(pow(np.sqrt(np.sum(np.multiply(dict_init,dict_init),axis=0)), -1))
    dict_init = dict_init.dot(temp)    
    basis_sign = np.sign(dict_init[0,:])
    dict_init = np.multiply(dict_init, basis_sign)

    print( 'Shape of dictionary : ' , str(dict_init.shape) + '\n')
    cv2.namedWindow('dict', cv2.WINDOW_NORMAL)
    cv2.imshow('dict',dict_init.astype('double'))

    return dict_init


def dict_update(D, data, matrix_sparse, atom_id):
    indices = np.where(matrix_sparse[atom_id, :] != 0)[0]
    D_temp = D
    sparse_temp = matrix_sparse[:,indices]

    if len(indices) > 1:
        sparse_temp[atom_id,:] = 0

        matrix_e_k = data[:, indices] - D_temp.dot(sparse_temp)
        u, s, v = svds(np.atleast_2d(matrix_e_k), 1)
        D_temp[:, atom_id] = u[:, 0]
        matrix_sparse[atom_id, indices] = s.dot(v)

    return D_temp, matrix_sparse

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- K-SVD ALGORITHM -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def k_svd(D, data):
    D_temp = D
                                                            # X(size mxp) = D(size mxn) x matrix_sparse(size nxp)
    matrix_sparse = np.zeros((D.T.dot(data)).shape)         # initializing spare matrix
    n_total = []
    num_iter = ksvd_iter
    print ('\nK-SVD, with residual criterion.')
    print ('-------------------------------')

    for k in range(num_iter):
        print ("Stage " , str(k+1) , "/" , str(num_iter) , "...")

        matrix_sparse = omp(D_temp, data)

        dict_elem_num = D.shape[1]
        count = 1

        for j in range(dict_elem_num):
            r = floor(count/float(dict_elem_num)*100)
            sys.stdout.write("\r- Dictionary updating : %d%%" % r)
            sys.stdout.flush()
            
            D_temp, matrix_sparse = dict_update(D_temp, data, matrix_sparse, j)
            count += 1
        print ('\r- Dictionary updating complete.\n')

    return D_temp, matrix_sparse, n_total


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ DENOISING METHOD -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def denoising(noisy_image):

    # 1. Form noisy patches.
    padded_noisy_image = pad(noisy_image, pad_width=window_shape, mode='symmetric')

    # dictionary intialization
    dict_init = dict_initiate(padded_noisy_image, window_stride)
 
    train_noisy_patches, train_noisy_patches_shape = patch_matrix_windows(padded_noisy_image, window_stride)
    train_data_mean = train_noisy_patches.mean()
    train_noisy_patches = train_noisy_patches - train_data_mean

    # 3. Compute K-SVD.
    start = timeit.default_timer()
    dict_final, sparse_init, n_total = k_svd(dict_init, train_noisy_patches)
    stop = timeit.default_timer()
    print ("Calculation time : " , str(stop - start) , ' seconds.')
    # print(np.max(dict_final), np.max(sparse_init))


    noisy_patches, noisy_patches_shape = patch_matrix_windows(padded_noisy_image, stride=1)
    data_mean = noisy_patches.mean()
    noisy_patches = noisy_patches - data_mean


    start = timeit.default_timer()
    sparse_final = omp(dict_final, noisy_patches)
    stop = timeit.default_timer()
    print ("Calculation time : " , str(stop - start) , ' seconds.')

    # 4. Reconstruct the image.
    patches_approx = dict_final.dot(sparse_final) + data_mean

    padded_denoised_image = reconstruct_image(padded_noisy_image.shape, patches_approx, padded_noisy_image)
    # patches_approx = patches_approx.reshape(noisy_patches.shape[1], *(patch_size,patch_size))
    # padded_denoised_image = reconstruct_from_patches_2d(patches_approx, (padded_noisy_image.shape[0]//2, padded_noisy_image.shape[1]//2))

    shrunk_0, shrunk_1 = tuple(map(sub, padded_denoised_image.shape, window_shape))
    denoised_image = np.abs(padded_denoised_image)[window_shape[0]:shrunk_0, window_shape[1]:shrunk_1]

    return denoised_image, stop - start, n_total



image = cv2.imread(args['image'], 0)

max_init_size = max(image.shape[0], image.shape[1])
resize_ratio = max_resize_dim/max_init_size

if resize_ratio < 1:
    image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)

noise_layer = np.random.normal(0, sigma ^ 2, image.size).reshape(image.shape).astype(int)
noisy_image = image + noise_layer

denoised_image, calc_time, n_total = denoising(noisy_image)

noisy_psnr = 20*log10(np.amax(image)) - 10*log10(pow(np.linalg.norm(image - noisy_image), 2)/noisy_image.size)
# diff_psnr = 20*log10(np.amax(noisy_image)) - 10*log10(pow(np.linalg.norm(noisy_image - denoised_image), 2)/denoised_image.size)
final_psnr = 20*log10(np.amax(image)) - 10*log10(pow(np.linalg.norm(image - denoised_image), 2)/denoised_image.size)
print(noisy_psnr, final_psnr)

cv2.namedWindow('orignal', cv2.WINDOW_NORMAL)
cv2.imshow('orignal', image)
cv2.namedWindow('noisy_image', cv2.WINDOW_NORMAL)
cv2.imshow('noisy_image', noisy_image.astype('uint8'))
cv2.namedWindow('denoised_image', cv2.WINDOW_NORMAL)
cv2.imshow('denoised_image', denoised_image.astype('uint8'))

name = ''

# cv2.imwrite(name + '1 - Greysc image.jpg', Image.fromarray(np.uint8(image)))
cv2.imwrite(name + '2 - Noisy image.jpg', noisy_image.astype('uint8'))

cv2.imwrite(name + '3 - Out - Step ' + str(window_stride) + ' - kSVD ' + str(ksvd_iter) +
       ' - Ratio ' + str(dict_ratio) + '.jpg', denoised_image.astype('uint8'))
cv2.imwrite(name + '4 - Difference - Step ' + str(window_stride) + ' - kSVD ' + str(ksvd_iter) +
       ' - Ratio ' + str(dict_ratio) + '.jpg', np.abs(noisy_image - denoised_image).astype('uint8'))


# cv2.waitKey(0)
while cv2.waitKey(-1) == 27:
    print('break')
    break
